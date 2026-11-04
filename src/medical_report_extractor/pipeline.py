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


def _repair_json_string(s: str) -> str:
    """Repair common LLM JSON syntax issues (unquoted ranges, double-escaped quotes, trailing commas, unclosed brackets)."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # 1. Normalize double-escaped quotes \" -> "
    s = s.replace(r'\"', '"')

    # 2. Quote unquoted numeric ranges like : 11.0-15.5 or : 13-43.0
    s = re.sub(r':\s*([0-9]+(?:\.[0-9]+)?\s*-\s*[0-9]+(?:\.[0-9]+)?)(\s*[,}\]])', r': "\1"\2', s)
    s = re.sub(r':\s*([0-9]+(?:\.[0-9]+)?\s*-\s*[0-9]+(?:\.[0-9]+)?)\s*$', r': "\1"', s)

    # 3. Quote comparison values like : <10 or : >50
    s = re.sub(r':\s*([<>]=?\s*[0-9]+(?:\.[0-9]+)?)(\s*[,}\]])', r': "\1"\2', s)
    s = re.sub(r':\s*([<>]=?\s*[0-9]+(?:\.[0-9]+)?)\s*$', r': "\1"', s)

    # 4. Remove trailing dangling commas
    s = re.sub(r',\s*$', '', s)
    s = re.sub(r',\s*([}\]])', r'\1', s)

    # 5. Track quote state and bracket stack for graceful repair
    in_string = False
    escape = False
    stack: list[str] = []

    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif ch in ('}', ']') and stack and stack[-1] == ch:
                stack.pop()

    if in_string:
        s += '"'

    s = re.sub(r',\s*$', '', s)

    # 6. Close any open structures in LIFO order
    while stack:
        s += stack.pop()

    return s


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

    # 3. Apply auto-repair (handles unquoted ranges like 11.0-15.5, escaped quotes, truncated brackets)
    try:
        repaired = _repair_json_string(cleaned)
        val = json.loads(repaired)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    # 4. Fallback: slice between outermost brackets
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            repaired_slice = _repair_json_string(cleaned[start : end + 1])
            val = json.loads(repaired_slice)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

    raise ValueError(f"Model response did not contain a valid JSON object. Response starts: {text[:300]!r}")


def _read_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _system_prompt(task_prompt: str) -> str:
    return f"{_read_prompt('shared.txt')}\n\n{_read_prompt(task_prompt)}"


def _parse_range_bounds(ref_str: str) -> tuple[float | None, float | None]:
    """Parse reference range string like '130-145', '11.0-15.5', '< 10', '> 50' into (low, high)."""
    if not ref_str:
        return None, None
    s = str(ref_str).strip()
    # Match standard range: "130-145" or "130 - 145" or "11.0 - 15.5"
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)$", s)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            return None, None
    # Match "< X" or "<= X"
    m_lt = re.match(r"^<=\s*([0-9]+(?:\.[0-9]+)?)$", s) or re.match(r"^<\s*([0-9]+(?:\.[0-9]+)?)$", s)
    if m_lt:
        try:
            return None, float(m_lt.group(1))
        except ValueError:
            return None, None
    # Match "> X" or ">= X"
    m_gt = re.match(r"^>=\s*([0-9]+(?:\.[0-9]+)?)$", s) or re.match(r"^>\s*([0-9]+(?:\.[0-9]+)?)$", s)
    if m_gt:
        try:
            return float(m_gt.group(1)), None
        except ValueError:
            return None, None
    return None, None


def _evaluate_numeric_status(val: float, low: float | None, high: float | None) -> tuple[str, str]:
    """
    Deterministically compare numeric value against reference range.
    Returns (status, urgency):
      - status: 'normal', 'high', 'low', 'critical_high', 'critical_low'
      - urgency: 'normal', 'mild', 'significant', 'critical'
    """
    if low is not None and high is not None:
        span = high - low
        if span <= 0:
            span = max(abs(low), 1.0)

        if val < low:
            deficit = low - val
            if deficit > 0.3 * span:
                return "critical_low", "critical"
            elif deficit > 0.1 * span:
                return "low", "significant"
            else:
                return "borderline_low", "mild"
        elif val > high:
            surplus = val - high
            if surplus > 0.3 * span:
                return "critical_high", "critical"
            elif surplus > 0.1 * span:
                return "high", "significant"
            else:
                return "borderline_high", "mild"
        else:
            return "normal", "normal"
    elif low is not None:  # e.g., > 50
        if val < low:
            return "low", "significant"
        return "normal", "normal"
    elif high is not None:  # e.g., < 10
        if val > high:
            return "high", "significant"
        return "normal", "normal"

    return "normal", "normal"


DEMOGRAPHIC_PREFIXES = (
    "age", "sex", "gender", "dob", "birth", "date", "name", "patient", "contact",
    "phone", "mobile", "address", "doctor", "dr", "ref", "hospital", "lab",
    "facility", "b.no", "bno", "bill", "mrn", "uhid", "ip", "op", "department"
)

def _is_demographic_key(key: str) -> bool:
    kl = str(key).lower().strip()
    if not kl or kl in {"parameter", "parameter_raw", "value", "value_raw", "value_numeric",
                       "unit", "unit_raw", "reference_range", "reference_range_raw", "flag"}:
        return True
    for prefix in DEMOGRAPHIC_PREFIXES:
        if kl == prefix or kl.startswith(prefix + "_") or kl.endswith("_" + prefix):
            return True
    return False

def _expand_flat_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    If a model put multiple lab parameters or metadata into a single flat dict
    (e.g., {'Sodium': 138, 'Potassium': 5, 'Sodium Level': '130-145', 'Age': 38, ...}),
    expand it into distinct individual items so they can be processed cleanly.
    """


    expanded: list[dict[str, Any]] = []
    for item in items:
        cat = str(item.get("category") or "").lower()
        # NEVER convert patient, provider, facility, or metadata items into investigations!
        if cat in {"patient", "provider", "facility", "document_metadata"}:
            expanded.append(item)
            continue

        data = item.get("data")
        if not isinstance(data, dict):
            expanded.append(item)
            continue

        # Check if data contains multiple key-value pairs representing tests
        # e.g., 'Sodium': 138, 'Potassium': 5, 'Chloride': 113
        test_keys = [
            k for k in data.keys()
            if not k.lower().endswith(" level")
            and not k.lower().endswith(" range")
            and not k.lower().endswith(" unit")
            and not _is_demographic_key(k)
        ]

        # If data only has standard fields or 1 test, leave it
        if len(test_keys) <= 1:
            expanded.append(item)
            continue

        # It's a bundled object. Unpack it!
        item_id = item.get("item_id", "ITEM_00000")
        source = item.get("source", {})
        sub_counter = 1

        for tk in test_keys:
            val = data[tk]
            # Look for matching range/level key, e.g. "Sodium Level" or "Sodium Range"
            ref_range = (
                data.get(f"{tk} Level")
                or data.get(f"{tk} level")
                or data.get(f"{tk} Range")
                or data.get(f"{tk} range")
                or data.get(f"{tk} Ref")
                or ""
            )
            unit_val = data.get(f"{tk} Unit") or data.get(f"{tk} unit") or ""

            try:
                num_val = float(str(val).strip())
            except (ValueError, TypeError):
                num_val = None

            new_item = {
                "item_id": f"{item_id}_{sub_counter}",
                "category": "investigation",
                "data": {
                    "parameter": tk,
                    "value_raw": str(val),
                    "value_numeric": num_val,
                    "unit_raw": str(unit_val),
                    "reference_range_raw": str(ref_range),
                    "flag": None,
                },
                "assertion_status": "present",
                "source": source,
                "extraction_status": "extracted",
            }
            expanded.append(new_item)
            sub_counter += 1

        # Also preserve patient/metadata fields if present
        meta_data: dict[str, Any] = {}
        for k, v in data.items():
            if k.lower() in {"age", "sex", "gender", "refby", "ref_by", "date", "name"}:
                meta_data[k] = v
        if meta_data:
            expanded.append({
                "item_id": f"{item_id}_meta",
                "category": "patient",
                "data": meta_data,
                "assertion_status": "present",
                "source": source,
                "extraction_status": "extracted",
            })

    return expanded

    return [items[i : i + size] for i in range(0, len(items), size)]


def _page_categories(page_map: PageMap) -> set[str]:
    cats = set()
    for section in page_map.sections:
        category = getattr(section, "category", None)
        if isinstance(category, str) and category in AGENTS:
            cats.add(category)
        elif isinstance(category, dict):
            # Fallback if model returned a nested dict for category
            cat_name = category.get("category") or category.get("name")
            if isinstance(cat_name, str) and cat_name in AGENTS:
                cats.add(cat_name)
    return cats


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

    def _interpret(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Stage 6: Semantic clinical interpretation — flag abnormal values, correlations, and risks."""
        if not items:
            return {
                "report_summary": "No extracted records available for interpretation.",
                "abnormal_values": [],
                "normal_values": [],
                "clinical_correlations": [],
                "risk_flags": [],
                "patient_alert": "No data was extracted from the document.",
            }

        # 1. First, compute deterministic range checks directly on items with values and reference ranges
        deterministic_abnormals: list[dict[str, Any]] = []
        deterministic_normals: list[dict[str, Any]] = []
        NON_CLINICAL_PARAMS = {"age", "sex", "gender", "refby", "ref_by", "date", "name", "b.no", "mrn", "contact", "phone", "address", "department", "uhid", "opd", "ipd", "mobile", "doctor", "dr", "hospital", "facility", "bill", "bno"}

        for item in items:
            cat = str(item.get("category") or "").lower()
            if cat in {"patient", "provider", "facility", "document_metadata"}:
                continue
            data = item.get("data", {})
            param = str(data.get("parameter") or "").strip()
            if not param or _is_demographic_key(param):
                continue

            ref_str = str(data.get("reference_range_raw") or data.get("reference_range") or "").strip()
            val_raw = data.get("value_raw") or data.get("value_numeric")
            if val_raw is None:
                continue

            try:
                num_val = float(str(val_raw).strip())
            except (ValueError, TypeError):
                continue

            low, high = _parse_range_bounds(ref_str)
            if low is None and high is None:
                continue

            status, urgency = _evaluate_numeric_status(num_val, low, high)
            unit_str = data.get("unit_raw") or ""
            val_display = f"{num_val} {unit_str}".strip()

            if status == "normal":
                deterministic_normals.append({
                    "item_id": item.get("item_id", ""),
                    "parameter": param,
                    "value": val_display,
                    "reference_range": ref_str,
                    "status": "normal",
                    "note": f"Within reference range ({ref_str}).",
                })
            else:
                direction = "high" if "high" in status else "low"
                sig = f"{param} is {status.replace('_', ' ')} (measured {val_display}, normal range is {ref_str})."
                deterministic_abnormals.append({
                    "item_id": item.get("item_id", ""),
                    "parameter": param,
                    "value": val_display,
                    "reference_range": ref_str,
                    "status": status,
                    "significance": sig,
                    "urgency": urgency,
                })

        # 2. Query model for high-level semantic synthesis (summary, correlations, risks, patient alert, potential conditions)
        try:
            response = self.client.complete(
                _system_prompt("interpret.txt"),
                "Interpret the following extracted medical report data. "
                "Provide entity-by-entity breakdowns (about, range, fetched value, classification) and "
                "suggest potential non-definitive health conditions or patterns to discuss with a physician.\n"
                + json.dumps(items, ensure_ascii=False),
            )
            result = _json_from_text(response)
        except Exception:
            result = {}

        result.setdefault("report_summary", "")
        result.setdefault("clinical_correlations", [])
        result.setdefault("risk_flags", [])
        result.setdefault("patient_alert", "")
        result.setdefault("potential_conditions", [])
        result.setdefault("analyzed_parameters", [])

        # 3. Clean up and reconcile abnormal values:
        # Filter out any non-clinical parameters (Age, Sex, RefBy, Date) that the model mistakenly flagged
        raw_abnormals = result.get("abnormal_values", [])
        model_abnormals = []
        for av in raw_abnormals:
            p_name = str(av.get("parameter", "")).strip().lower()
            if not p_name or _is_demographic_key(p_name):
                continue
            # If status is normal, skip it from abnormal list
            if av.get("status", "").lower() == "normal" or av.get("urgency", "").lower() == "normal":
                continue
            model_abnormals.append(av)

        # Merge deterministic calculations with model results
        # Deterministic math takes priority for status/urgency of numerical values!
        det_by_param = {a["parameter"].lower(): a for a in deterministic_abnormals}
        final_abnormals: list[dict[str, Any]] = []

        # Add all verified deterministic abnormals
        for det_a in deterministic_abnormals:
            # Check if model gave a more descriptive clinical explanation
            matching_model = next((m for m in model_abnormals if m.get("parameter", "").lower() == det_a["parameter"].lower()), None)
            if matching_model and matching_model.get("significance"):
                det_a["significance"] = matching_model["significance"]
            final_abnormals.append(det_a)

        # Add any qualitative model abnormals (e.g. imaging, ecg findings) that weren't numerical ranges
        det_params_lower = {a["parameter"].lower() for a in deterministic_abnormals}
        for ma in model_abnormals:
            if ma.get("parameter", "").lower() not in det_params_lower and ma.get("parameter", "").lower() not in {n["parameter"].lower() for n in deterministic_normals}:
                final_abnormals.append(ma)

        result["abnormal_values"] = final_abnormals

        # Reconcile normal values
        det_normals_by_param = {n["parameter"].lower(): n for n in deterministic_normals}
        raw_normals = result.get("normal_values", [])
        final_normals = list(deterministic_normals)
        for rn in raw_normals:
            p_name = str(rn.get("parameter", "")).strip().lower()
            if p_name and not _is_demographic_key(p_name) and p_name not in det_normals_by_param and p_name not in det_params_lower:
                final_normals.append(rn)
        result["normal_values"] = final_normals

        # 4. Synthesize comprehensive entity-by-entity analyzed_parameters
        raw_params = result.get("parameters") or result.get("analyzed_parameters") or []
        model_analyzed = {str(ap.get("parameter", "")).strip().lower(): ap for ap in raw_params if isinstance(ap, dict)}
        final_analyzed_params: list[dict[str, Any]] = []
        seen_analyzed: set[str] = set()

        for item in items:
            cat = str(item.get("category") or "").lower()
            if cat in {"patient", "provider", "facility", "document_metadata"}:
                continue
            data = item.get("data", {})
            param = str(data.get("parameter") or "").strip()
            if not param or _is_demographic_key(param) or param.lower() in seen_analyzed:
                continue
            seen_analyzed.add(param.lower())

            val_raw = data.get("value_raw") or data.get("value_numeric")
            unit_str = data.get("unit_raw") or ""
            val_display = f"{val_raw} {unit_str}".strip() if val_raw is not None else "Not reported"
            ref_str = str(data.get("reference_range_raw") or data.get("reference_range") or "").strip() or "Standard reference interval"

            # Check if this item is in abnormals or normals
            matching_ab = next((a for a in final_abnormals if a["parameter"].lower() == param.lower()), None)
            matching_norm = next((n for n in final_normals if n["parameter"].lower() == param.lower()), None)

            if matching_ab:
                classification = matching_ab.get("status", "high")
                if "high" in classification:
                    classification = "critical_high" if "critical" in classification else "high"
                elif "low" in classification:
                    classification = "critical_low" if "critical" in classification else "low"
                clin_sig = matching_ab.get("significance", f"Value is {classification.replace('_', ' ')} relative to reference range.")
            elif matching_norm:
                classification = "in_range"
                clin_sig = matching_norm.get("note", "Result is within physiological reference range.")
            else:
                classification = "in_range"
                clin_sig = "Parameter observed in report."

            # Check if model provided a rich "about" explanation
            model_info = model_analyzed.get(param.lower(), {})
            about_text = model_info.get("about") or f"{param} is a clinical diagnostic marker measured to assess bodily organ health and metabolic balance."

            final_analyzed_params.append({
                "item_id": item.get("item_id", ""),
                "parameter": param,
                "about": about_text,
                "fetched_value": val_display,
                "normal_range": ref_str,
                "classification": classification,
                "clinical_significance": clin_sig,
            })

        result["analyzed_parameters"] = final_analyzed_params

        # 5. Ensure potential_conditions has prudent, non-certain phrasing and fallback if empty
        pot_conds = result.get("potential_conditions") or result.get("possible_conditions") or []
        valid_conds: list[dict[str, Any]] = []
        for pc in pot_conds:
            if isinstance(pc, dict) and pc.get("condition"):
                pc.setdefault("caution_note", "This is a potential consideration for review with your doctor, NOT a definitive diagnosis.")
                valid_conds.append(pc)

        # Fallback if model did not return potential conditions but there are abnormal findings
        if not valid_conds and final_abnormals:
            ab_names = [a["parameter"] for a in final_abnormals]
            valid_conds.append({
                "condition": f"Clinical follow-up for altered {', '.join(ab_names[:3])}",
                "associated_findings": ab_names,
                "explanation": f"The documented variations in {', '.join(ab_names)} warrant evaluation to assess underlying metabolic or systemic factors.",
                "caution_note": "These are potential considerations for review with your doctor, NOT a confirmed diagnosis."
            })
        result["potential_conditions"] = valid_conds

        # Fallback patient alert and summary if empty
        if not result["report_summary"]:
            if final_abnormals:
                ab_names = ", ".join(a["parameter"] for a in final_abnormals[:4])
                result["report_summary"] = f"Diagnostic report indicates out-of-range findings for {ab_names}."
            else:
                result["report_summary"] = "All evaluated diagnostic parameters fall within standard reference intervals."

        if not result["patient_alert"]:
            if final_abnormals:
                ab_names = ", ".join(a["parameter"] for a in final_abnormals[:3])
                result["patient_alert"] = f"Your lab results showed abnormal levels for {ab_names}. Please discuss these results with your attending physician to evaluate the necessary next steps."
            else:
                result["patient_alert"] = "Your test results are within normal limits. Discuss these findings with your doctor at your next scheduled visit."

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
                if isinstance(omission, dict) and omission.get("category") in AGENTS and omission.get("confidence") in {"high", "medium"}
            ]
            if not candidates:
                break
            retried_keys: set[tuple[int, str, str]] = set()
            retry_additions: list[dict[str, Any]] = []
            for page_number, omission in candidates:
                desc = omission.get("description")
                if isinstance(desc, dict):
                    desc = json.dumps(desc)
                key = (int(page_number), str(omission.get("category")), str(desc))
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
                touched_pages = {int(entry["page"]) for entry in retry_log if "page" in entry and entry.get("page") is not None}
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

        # Ensure any bundled flat objects from local model are unpacked into individual items
        extracted_items = _expand_flat_items(extracted_items)

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

        # Stage 6: Semantic clinical interpretation.
        try:
            interpretation = self._interpret(extracted_items)
        except Exception as exc:
            interpretation = {
                "report_summary": f"Interpretation unavailable: {exc}",
                "abnormal_values": [],
                "normal_values": [],
                "clinical_correlations": [],
                "risk_flags": [],
                "patient_alert": "Clinical interpretation could not be generated.",
            }

        final_data = {
            "document_id": document_id,
            "source_file_name": input_path.name,
            "document_map": document_map,
            "items": extracted_items,
            "reconciliation": reconciliation,
            "interpretation": interpretation,
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

        page_images = [page.image_data_url for page in pages]
        generate_dashboard_html(final_data, audit_data, page_images, dashboard_path)
        return report_path, audit_path
