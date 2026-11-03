from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    model_config = ConfigDict(extra="allow")
    category: Literal["patient", "clinical", "investigations", "medications", "care", "other"]
    description: str = ""
    region_hint: str = ""


class PageMap(BaseModel):
    model_config = ConfigDict(extra="allow")
    page_number: int = Field(ge=1)
    document_type: str = "unknown"
    sections: list[Section] = Field(default_factory=list)
    unreadable_regions: list[str] = Field(default_factory=list)
    page_notes: list[str] = Field(default_factory=list)


class Source(BaseModel):
    model_config = ConfigDict(extra="allow")
    document_id: str
    page: int | None = Field(default=None, ge=1)
    evidence: str = ""
    region: str | None = None


class ExtractedItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    item_id: str
    category: str
    data: dict[str, Any] = Field(default_factory=dict)
    assertion_status: Literal["present", "absent", "uncertain", "unknown"] = "unknown"
    source: Source
    extraction_status: Literal["extracted", "ambiguous", "illegible", "needs_review"] = "extracted"


def validate_extracted_item(raw: dict[str, Any], document_id: str, item_id: str) -> dict[str, Any]:
    """Normalize the common record envelope while preserving category-specific data."""
    source_raw = raw.get("source") or {}
    source = {
        "document_id": document_id,
        "page": source_raw.get("page"),
        "evidence": str(source_raw.get("evidence") or ""),
        "region": source_raw.get("region"),
    }
    candidate = {
        "item_id": item_id,
        "category": str(raw.get("category") or "other"),
        "data": raw.get("data") if isinstance(raw.get("data"), dict) else {},
        "assertion_status": raw.get("assertion_status", "unknown"),
        "source": source,
        "extraction_status": raw.get("extraction_status", "needs_review"),
    }
    try:
        return ExtractedItem.model_validate(candidate).model_dump()
    except Exception as exc:
        # Preserve the raw candidate for audit rather than dropping a model response.
        candidate["extraction_status"] = "needs_review"
        candidate["validation_error"] = str(exc)
        if candidate["assertion_status"] not in {"present", "absent", "uncertain", "unknown"}:
            candidate["assertion_status"] = "unknown"
        if candidate["extraction_status"] not in {"extracted", "ambiguous", "illegible", "needs_review"}:
            candidate["extraction_status"] = "needs_review"
        return candidate
