from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz
from PIL import Image


@dataclass
class DocumentPage:
    page_number: int
    text: str
    image_data_url: str


def _to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def load_document(path: str | Path, render_dpi: int = 144) -> list[DocumentPage]:
    """Load a PDF or image into page text plus image data URLs for VLM input."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    suffix = file_path.suffix.lower()
    pages: list[DocumentPage] = []

    if suffix == ".pdf":
        pdf = fitz.open(file_path)
        try:
            scale = render_dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            for index, page in enumerate(pdf, start=1):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_bytes = pixmap.tobytes("jpeg")
                pages.append(
                    DocumentPage(
                        page_number=index,
                        text=page.get_text("text") or "",
                        image_data_url=_to_data_url(image_bytes),
                    )
                )
        finally:
            pdf.close()
        return pages

    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=92)
            image_bytes = buffer.getvalue()
        return [DocumentPage(page_number=1, text="", image_data_url=_to_data_url(image_bytes))]

    raise ValueError(f"Unsupported input format '{suffix}'. Use PDF, PNG, JPG/JPEG, or WEBP.")
