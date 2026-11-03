from __future__ import annotations

import base64
import binascii
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

if TYPE_CHECKING:
    from .documents import DocumentPage


class LocalQwenClient:
    """Run a local Qwen3.5 vision-language checkpoint through Transformers."""

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 4096,
        device_map: str = "auto",
        local_files_only: bool = True,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local model directory does not exist: {self.model_path}. "
                "Set MODEL_PATH in .env to the directory created by snapshot_download()."
            )
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"config.json was not found in {self.model_path}. "
                "MODEL_PATH must point to the downloaded model directory."
            )

        self.model_name = self.model_path.name
        self.max_new_tokens = max_new_tokens

        try:
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                local_files_only=local_files_only,
            )
            self.model = AutoModelForMultimodalLM.from_pretrained(
                str(self.model_path),
                device_map=device_map,
                local_files_only=local_files_only,
            )
        except (ImportError, ValueError, OSError) as exc:
            raise RuntimeError(
                "Could not load Qwen3.5 with the installed Transformers stack. "
                "Install a recent Transformers version plus Accelerate, and check "
                "that the local model snapshot is complete. Original error: "
                f"{exc}"
            ) from exc

        self.model.eval()

    @staticmethod
    def _write_data_url(data_url: str, destination: Path) -> None:
        if not data_url.startswith("data:") or ";base64," not in data_url:
            raise ValueError("Document page image must be a base64 data URL.")
        encoded = data_url.split(";base64,", 1)[1]
        try:
            destination.write_bytes(base64.b64decode(encoded, validate=True))
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Could not decode the rendered page image.") from exc

    def complete(
        self,
        system_prompt: str,
        user_text: str,
        pages: list[DocumentPage] | None = None,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": user_text}]

        # Transformers' multimodal processor accepts local image paths. Temporary
        # files keep rendered patient pages out of the repository and are removed
        # immediately after the model call.
        with tempfile.TemporaryDirectory(prefix="medical_report_pages_") as temp_dir:
            temp_root = Path(temp_dir)
            for index, page in enumerate(pages or []):
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"\n[Source page number: {page.page_number}]\n"
                            "Extracted page text:\n"
                            f"{page.text or '[No machine-readable text; inspect the page image.]'}"
                        ),
                    }
                )
                image_path = temp_root / f"page_{index + 1}.jpg"
                self._write_data_url(page.image_data_url, image_path)
                content.append({"type": "image", "url": str(image_path)})

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {"role": "user", "content": content},
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            prompt_length = inputs["input_ids"].shape[1]
            generated_tokens = generated[:, prompt_length:]
            result = self.processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        if not result.strip():
            raise RuntimeError("The local Qwen model returned an empty response.")
        return result.strip()
