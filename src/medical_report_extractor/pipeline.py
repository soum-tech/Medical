from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .documents import DocumentPage, load_document
from .schemas import PageMap, validate_extracted_item
from .local_model import LocalQwenClient
from .dashboard import generate_dashboard_html

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "prompts"

AGENTS = {
    "patient": "extract_patient.txt",
    "clinical": "extract_clinical.txt",
    "investigations": "extract_investigations.txt",
    "medications": "extract_medications.txt",
    "care": "extract_care.txt",
}


def _json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    
    # 1. Fast path: direct parse
    try:
        val = json.loads(cleaned)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass

    # 2. Extract first valid balanced JSON object using raw_decode
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = cleaned.find("{", pos)
        if start == -1:
            break
        try:
            val, _ = decoder.raw_decode(cleaned, idx=start)
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass
        pos = start + 1

    # 3. Fallback: slice between outermost brackets
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            val = json.loads(cleaned[start : end + 1])
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Model response did not contain a valid JSON object. Response starts: {text[:300]!r}")


def _read_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _system_prompt(task_prompt: str) -> str:
    return f"{_read_prompt('shared.txt')}\n\n{_read_prompt(task_prompt)}"


def _chunks(items: list[DocumentPage], size: int) -> list[list[DocumentPage]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _page_categories(page_map: PageMap) -> set[str]:
    return {section.category for section in page_map.sections if section.category in AGENTS}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class ExtractionPipeline:
    def __init__(self) -> None:
        load_dotenv()
        backend = os.getenv("BACKEND", "local").lower().strip()

        if backend == "gemini":
            from .gemini_client import GeminiClient
            self.client = GeminiClient(
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                max_output_tokens=_env_int("GEMINI_MAX_OUTPUT_TOKENS", 4096),
            )
        else:
            model_path = Path(os.getenv("MODEL_PATH", "models/Qwen3.5-0.8B")).expanduser()
            if not model_path.is_absolute():
                model_path = ROOT / model_path
            self.client = LocalQwenClient(
                model_path=model_path,
                max_new_tokens=_env_int("MODEL_MAX_NEW_TOKENS", 4096),
                device_map=os.getenv("MODEL_DEVICE_MAP", "auto"),
                local_files_only=os.getenv("MODEL_LOCAL_FILES_ONLY", "true").lower() in {"1", "true", "yes"},
            )

        self.render_dpi = _env_int("RENDER_DPI", 144)
        self.pages_per_call = max(1, _env_int("MAX_PAGES_PER_CALL", 2))
        self.max_retries = max(0, _env_int("MAX_TARGETED_RETRIES", 1))
        self._item_counter = 0

    def _next_id(self) -> str:
        self._item_counter += 1
        return f"ITEM_{self._item_counter:05d}"

    def _map_pages(self, pages: list[DocumentPage]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        prompt = _system_prompt("map_document.txt")
        for page in pages:
            user_text = f"Map page {page.page_number}. Return the required page mapping JSON."
            try:
                raw = _json_from_text(self.client.complete(prompt, user_text, [page]))
                raw.setdefault("page_number", page.page_number)
                page_map = PageMap.model_validate(raw).model_dump()
            except Exception as exc:
                # Mapping failure must not cause a page to be skipped: route it to every extractor.
                page_map = PageMap(
                    page_number=page.page_number,
                    document_type="mapping_failed",
                    sections=[{"category": category, "description": "Fallback extraction because page mapping failed."} for category in AGENTS],
                    page_notes=[f"Page mapping failed; all extraction categories will run. Error: {exc}"],
                ).model_dump()
            mapped.append(page_map)
        return mapped

    def _extract_category(
        self,
        category: str,
        pages: list[DocumentPage],
        document_id: str,
        retry_context: str = "",
    ) -> tuple[list[dict[str, Any]], list[str]]:
        prompt = _system_prompt(AGENTS[category])
        records: list[dict[str, Any]] = []
        notes: list[str] = []
        for page_chunk in _chunks(pages, self.pages_per_call):
            page_numbers = [page.page_number for page in page_chunk]
            user_text = (
                f"Document ID: {document_id}\n"
                f"Target extraction category: {category}\n"
                f"Source pages in this call: {page_numbers}\n"
                "Extract all items belonging to this agent's scope from these pages. "
                "For each item, set source.page to the original page number.\n"
                f"Additional audit context (if any): {retry_context or 'None'}\n"
                "Return the JSON object specified by the task prompt."
            )
            response = self.client.complete(prompt, user_text, page_chunk)
            parsed = _json_from_text(response)
            raw_items = parsed.get("items", [])
            if not isinstance(raw_items, list):
                raise ValueError(f"Agent '{category}' returned 'items' in a non-list format.")
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                records.append(validate_extracted_item(raw_item, document_id, self._next_id()))
            if isinstance(parsed.get("page_notes"), list):
                notes.extend(str(note) for note in parsed["page_notes"])
        return records, notes

    def _reconcile(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"possible_duplicates": [], "possible_conflicts": [], "related_records": [], "notes": ["No extracted records to reconcile."]}
        # Do not send an unbounded payload to the model. Reconciliation is advisory only.
        response = self.client.complete(
            _system_prompt("reconcile.txt"),
            "Review these extracted records. Use only the supplied item IDs.\n" + json.dumps(items, ensure_ascii=False),
        )
        return _json_from_text(response)

    def _verify_page(
        self,
        page: DocumentPage,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        page_items = [item for item in items if item.get("source", {}).get("page") == page.page_number]
        user_text = (
            f"Audit source page {page.page_number}.\n"
            "Extracted records attributed to this page:\n"
            + json.dumps(page_items, ensure_ascii=False)
            + "\nReturn the completeness-audit JSON."
        )
        result = _json_from_text(self.client.complete(_system_prompt("verify.txt"), user_text, [page]))
        result.setdefault("page_number", page.page_number)
        result.setdefault("omissions", [])
        result.setdefault("potential_errors", [])
        return result

    def run(self, input_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        document_id = input_path.stem
        pages = load_document(input_path, self.render_dpi)

        # Stage 1: document map.
        document_map = self._map_pages(pages)

        # Stage 2: specialized extraction. Unmapped pages are sent to all extractors.
        page_map_by_number = {item["page_number"]: PageMap.model_validate(item) for item in document_map}
        extracted_items: list[dict[str, Any]] = []
        extraction_notes: list[dict[str, Any]] = []
        for category in AGENTS:
            selected_pages = [
                page for page in pages
                if category in _page_categories(page_map_by_number[page.page_number])
            ]
            # A mapper that finds no categories for a page is not allowed to suppress extraction.
            for page in pages:
                if not _page_categories(page_map_by_number[page.page_number]) and page not in selected_pages:
                    selected_pages.append(page)
            if selected_pages:
                records, notes = self._extract_category(category, selected_pages, document_id)
                extracted_items.extend(records)
                extraction_notes.append({"category": category, "notes": notes})

        # Stage 3: advisory reconciliation. Source records are always retained.
        try:
            reconciliation = self._reconcile(extracted_items)
        except Exception as exc:
            reconciliation = {"possible_duplicates": [], "possible_conflicts": [], "related_records": [], "notes": [f"Reconciliation unavailable: {exc}"]}

        # Stage 4: independent page-wise completeness audit.
        audit_pages: list[dict[str, Any]] = []
        for page in pages:
            try:
                audit_pages.append(self._verify_page(page, extracted_items))
            except Exception as exc:
                audit_pages.append({
                    "page_number": page.page_number,
                    "omissions": [],
                    "potential_errors": [],
                    "unreadable_or_ambiguous_regions": [],
                    "page_complete": False,
                    "audit_error": str(exc),
                })

        # Stage 5: targeted retries for verifier-identified omissions. Bounded by configuration.
        retry_log: list[dict[str, Any]] = []
        for retry_round in range(self.max_retries):
            candidates = [
                (audit.get("page_number"), omission)
                for audit in audit_pages
                for omission in audit.get("omissions", [])
                if omission.get("category") in AGENTS and omission.get("confidence") in {"high", "medium"}
            ]
            if not candidates:
                break
            retried_keys: set[tuple[int, str, str]] = set()
            retry_additions: list[dict[str, Any]] = []
            for page_number, omission in candidates:
                key = (int(page_number), str(omission.get("category")), str(omission.get("description")))
                if key in retried_keys:
                    continue
                retried_keys.add(key)
                source_page = next((page for page in pages if page.page_number == int(page_number)), None)
                if source_page is None:
                    continue
                category = str(omission["category"])
                try:
                    records, notes = self._extract_category(
                        category,
                        [source_page],
                        document_id,
                        retry_context=(
                            "The completeness auditor flagged this possible omission: "
                            + json.dumps(omission, ensure_ascii=False)
                            + ". Re-check the original page and extract only source-supported items."
                        ),
                    )
                    retry_additions.extend(records)
                    retry_log.append({"round": retry_round + 1, "page": page_number, "category": category, "finding": omission, "new_items": len(records), "notes": notes})
                except Exception as exc:
                    retry_log.append({"round": retry_round + 1, "page": page_number, "category": category, "finding": omission, "error": str(exc)})
            extracted_items.extend(retry_additions)
            if retry_additions:
                # Re-audit pages touched by retries so the audit reflects the updated extraction.
                touched_pages = {entry["page"] for entry in retry_log if "page" in entry}
                updated_audits = []
                for audit in audit_pages:
                    if audit.get("page_number") in touched_pages:
                        page = next(p for p in pages if p.page_number == audit["page_number"])
                        try:
                            updated_audits.append(self._verify_page(page, extracted_items))
                        except Exception as exc:
                            updated_audits.append({**audit, "re_audit_error": str(exc)})
                    else:
                        updated_audits.append(audit)
                audit_pages = updated_audits

        # Final validation: preserve questionable records and mark them for review rather than dropping.
        for item in extracted_items:
            source = item.get("source", {})
            page_number = source.get("page")
            if page_number is None or not any(page.page_number == page_number for page in pages):
                item["extraction_status"] = "needs_review"
                item.setdefault("validation_warnings", []).append("Missing or invalid source page reference.")
            if not source.get("evidence"):
                item["extraction_status"] = "needs_review"
                item.setdefault("validation_warnings", []).append("No supporting text evidence was returned.")

        # Reconciliation is rerun after targeted extraction so IDs in suggestions remain current.
        try:
            reconciliation = self._reconcile(extracted_items)
        except Exception as exc:
            reconciliation.setdefault("notes", []).append(f"Final reconciliation unavailable: {exc}")

        final_data = {
            "document_id": document_id,
            "source_file_name": input_path.name,
            "document_map": document_map,
            "items": extracted_items,
            "reconciliation": reconciliation,
            "pipeline_metadata": {
                "page_count": len(pages),
                "model": self.client.model_name,
                "extraction_categories": list(AGENTS.keys()),
                "targeted_retry_limit": self.max_retries,
            },
        }
        audit_data = {
            "document_id": document_id,
            "page_audits": audit_pages,
            "extraction_notes": extraction_notes,
            "targeted_retries": retry_log,
            "unresolved_omissions": [
                {"page": audit.get("page_number"), **omission}
                for audit in audit_pages
                for omission in audit.get("omissions", [])
            ],
            "unreadable_or_ambiguous_regions": [
                {"page": audit.get("page_number"), "region": region}
                for audit in audit_pages
                for region in audit.get("unreadable_or_ambiguous_regions", [])
            ],
        }

        report_path = output_dir / "medical_report.json"
        audit_path = output_dir / "extraction_audit.json"
        dashboard_path = output_dir / "dashboard.html"
        _write_json(report_path, final_data)
        _write_json(audit_path, audit_data)

        # Compute relative image path from output_dir to input_path
        try:
            rel_image_path = os.path.relpath(input_path, output_dir).replace("\\", "/")
        except Exception:
            rel_image_path = f"../input/{input_path.name}"

        generate_dashboard_html(final_data, audit_data, rel_image_path, dashboard_path)
        return report_path, audit_path
