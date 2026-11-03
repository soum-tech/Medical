from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

if TYPE_CHECKING:
    from .documents import DocumentPage


class GeminiClient:
    """Call the Gemini API instead of a local model.

    Drop-in replacement for LocalQwenClient. The pipeline calls
    ``client.complete(system_prompt, user_text, pages)`` and receives a
    plain-text string back; everything else in the pipeline is unchanged.

    Uses the modern ``google-genai`` SDK (v2+).
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        max_output_tokens: int = 4096,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError(
                "No Gemini API key found. Set GEMINI_API_KEY in your .env file."
            )
        self.model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._client = genai.Client(api_key=key)

    # ------------------------------------------------------------------
    # Public interface (matches LocalQwenClient)
    # ------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_text: str,
        pages: list[DocumentPage] | None = None,
    ) -> str:
        """Send a prompt + optional page images to Gemini and return raw text."""
        parts: list = [user_text]

        for page in pages or []:
            parts.append(
                f"\n[Source page number: {page.page_number}]\n"
                f"Extracted page text:\n"
                f"{page.text or '[No machine-readable text; inspect the page image.]'}"
            )
            image_bytes = _data_url_to_bytes(page.image_data_url)
            parts.append(
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=self._max_output_tokens,
                temperature=0.0,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )

        text = response.text
        if not text or not text.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return text.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_url_to_bytes(data_url: str) -> bytes:
    """Decode a base64 data URL to raw bytes."""
    if ";base64," not in data_url:
        raise ValueError("Page image is not a base64 data URL.")
    encoded = data_url.split(";base64,", 1)[1]
    return base64.b64decode(encoded)
