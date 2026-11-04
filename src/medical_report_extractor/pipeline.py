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
      - urgency: 'normal', 'significant', 'critical'
    """
    if low is not None and high is not None:
        span = high - low
        if span <= 0:
            span = max(abs(low), 1.0)

        if val < low:
            deficit = low - val
            if deficit > 0.3 * span:
                return "critical_low", "critical"
            return "low", "significant"
        elif val > high:
            surplus = val - high
            if surplus > 0.3 * span:
                return "critical_high", "critical"
            return "high", "significant"
        else:
            return "normal", "normal"
    elif low is not None:
        if val < low:
            return "low", "significant"
        return "normal", "normal"
    elif high is not None:
        if val > high:
            return "high", "significant"
        return "normal", "normal"

    return "normal", "normal"


DEMOGRAPHIC_PREFIXES = (
    "age", "sex", "gender", "dob", "birth", "date", "name", "patient", "contact",
    "phone", "mobile", "address", "doctor", "dr", "ref", "hospital", "facility",
    "bno", "bill", "mrn", "uhid", "consultant", "pathologist", "specimen"
)


def _is_demographic_key(key: str) -> bool:
    kl = re.sub(r"[^a-z0-9]", "", str(key).lower())
    if not kl or kl in {
        "parameter", "value", "valueraw", "valuenumeric",
        "unit", "unitraw", "referencerange", "referencerangeraw", "flag"
    }:
        return True
    return any(kl.startswith(p) for p in DEMOGRAPHIC_PREFIXES)


MEDICAL_TEST_KNOWLEDGE = {
    "sodium": "Sodium is an essential electrolyte that maintains fluid balance, blood pressure, and cellular nerve and muscle function.",
    "potassium": "Potassium is a vital electrolyte that regulates heart rhythm, fluid balance, and electrical signals in nerve and muscle cells.",
    "chloride": "Chloride is an essential electrolyte that collaborates with sodium and potassium to regulate hydration and acid-base (pH) balance.",
    "bicarbonate": "Bicarbonate is an electrolyte and key chemical buffer that regulates systemic acid-base (pH) balance in the blood.",
    "urea": "Urea is a metabolic byproduct of protein breakdown filtered and excreted by the kidneys, used to evaluate renal function.",
    "creatinine": "Creatinine is a waste product of muscle catabolism filtered out by healthy kidneys, serving as a primary marker of kidney filtration health.",
    "uric acid": "Uric acid is a waste product from purine metabolism, evaluated to assess kidney excretory capacity and risks for gout or renal calculi.",
    "haemoglobin": "Haemoglobin is an iron-rich protein in red blood cells responsible for carrying oxygen from the lungs to organs and tissues throughout the body.",
    "hemoglobin": "Hemoglobin is an iron-rich protein in red blood cells responsible for carrying oxygen from the lungs to organs and tissues throughout the body.",
    "glucose": "Glucose is the body's primary metabolic fuel source, tightly regulated by insulin to assess glycemic control, diabetes risk, or hypoglycemia.",
    "platelets": "Platelets are blood cells that initiate clotting and vascular repair to prevent excessive bleeding.",
    "wbc": "White blood cells are immune cells that protect against microbial infections, inflammatory processes, and systemic illness.",
    "rbc": "Red blood cells transport oxygen and carbon dioxide between the lungs and peripheral body tissues.",
    "bilirubin": "Bilirubin is a yellowish compound formed during the breakdown of red blood cells, processed by the liver to assess biliary and hepatic health.",
    "albumin": "Albumin is a liver-synthesized plasma protein that maintains vascular oncotic pressure and transports hormones, vitamins, and medications.",
    "total protein": "Total protein measures the combined amount of albumin and globulins to evaluate liver function, nutritional status, and kidney health.",
    "calcium": "Calcium is an essential mineral required for bone structure, cardiac contractions, nerve signaling, and blood coagulation.",
    "cholesterol": "Cholesterol is a lipid substance essential for building cell membranes, synthesizing vitamin D, and producing steroid hormones.",
    "triglycerides": "Triglycerides are the main form of stored body fat, utilized for cellular energy and evaluated as a marker of cardiovascular risk.",
}


def _get_medical_test_about(param: str) -> str:
    p_lower = param.lower().strip()
    if p_lower in MEDICAL_TEST_KNOWLEDGE:
        return MEDICAL_TEST_KNOWLEDGE[p_lower]
    for k, desc in MEDICAL_TEST_KNOWLEDGE.items():
        if k in p_lower:
            return desc
    return f"{param} is a clinical diagnostic marker measured to evaluate specific organ function, metabolic equilibrium, and physiological balance."


def _expand_flat_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    If a model put multiple lab parameters or metadata into a single flat dict
    (e.g., {'Sodium': 138, 'Potassium': 5, 'Sodium Level': '130-145', 'Age': 38, ...}),
    expand it into distinct individual items so they can be processed cleanly.
    """
    expanded: list[dict[str, Any]] = []
    for item in items:
        cat = str(item.get("category") or "").lower()
        if any(p in cat for p in ("patient", "provider", "facility", "metadata", "demographic")):
            expanded.append(item)
            continue

        data = item.get("data")
        if not isinstance(data, dict):
            expanded.append(item)
            continue

        test_keys = [
            k for k in data.keys()
            if not k.lower().endswith(" level")
            and not k.lower().endswith(" range")
            and not k.lower().endswith(" unit")
            and not _is_demographic_key(k)
        ]

        if len(test_keys) <= 1:
            expanded.append(item)
            continue

        item_id = item.get("item_id", "ITEM_00000")
        source = item.get("source", {})
        sub_counter = 1

        for tk in test_keys:
            val = data[tk]
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
                m_num = re.search(r"[-+]?\d*\.?\d+", str(val))
                num_val = float(m_num.group(0)) if m_num else None

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

        meta_data: dict[str, Any] = {}
        for k, v in data.items():
            if _is_demographic_key(k):
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


def _chunks(lst: list[Any], n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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

    def _parse_category_items(
        self,
        parsed: Any,
        category: str,
        document_id: str,
        default_page: int,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        if category == "patient":
            patient_data: dict[str, Any] = {}
            if isinstance(parsed, dict):
                if "patient" in parsed and isinstance(parsed["patient"], dict):
                    patient_data = parsed["patient"]
                elif "items" in parsed and isinstance(parsed["items"], list):
                    for it in parsed["items"]:
                        if isinstance(it, dict):
                            d = it.get("data") if isinstance(it.get("data"), dict) else {k: v for k, v in it.items() if k not in {"category", "source", "assertion_status"}}
                            patient_data.update(d)
                else:
                    patient_data = {k: v for k, v in parsed.items() if k not in {"items", "page_notes"}}

            if patient_data:
                clean_patient = {k: v for k, v in patient_data.items() if _is_demographic_key(k) and v is not None and str(v).strip()}
                if clean_patient:
                    records.append({
                        "item_id": self._next_id(),
                        "category": "patient",
                        "data": clean_patient,
                        "assertion_status": "present",
                        "source": {"document_id": document_id, "page": default_page, "evidence": "Patient demographic header"},
                        "extraction_status": "extracted",
                    })
            return records

        if category == "investigations":
            raw_list = []
            if isinstance(parsed, dict):
                raw_list = (
                    parsed.get("items")
                    or parsed.get("tests")
                    or parsed.get("investigations")
                    or parsed.get("results")
                    or parsed.get("parameters")
                    or []
                )
                if not raw_list:
                    for k, v in parsed.items():
                        if (k not in {"page_notes", "sections"} and not _is_demographic_key(k) and
                            not k.lower().endswith(" level") and not k.lower().endswith(" range") and not k.lower().endswith(" unit")):
                            ref = parsed.get(f"{k} Level") or parsed.get(f"{k} Range") or parsed.get(f"{k} Ref") or ""
                            unit = parsed.get(f"{k} Unit") or parsed.get(f"{k} unit") or ""
                            raw_list.append({"name": k, "value": str(v), "normal_range": str(ref), "unit": str(unit)})
            elif isinstance(parsed, list):
                raw_list = parsed

            for entry in raw_list:
                if not isinstance(entry, dict):
                    continue
                d = entry.get("data") if isinstance(entry.get("data"), dict) else entry
                p_name = (
                    d.get("name")
                    or d.get("parameter")
                    or d.get("test")
                    or d.get("test_name")
                    or d.get("investigation")
                    or ""
                )
                p_name = str(p_name).strip()

                if not p_name:
                    cand_keys = [k for k in d.keys() if not _is_demographic_key(k) and k not in {"category", "source", "assertion_status", "extraction_status", "page"}]
                    if len(cand_keys) == 1:
                        p_name = cand_keys[0]
                        val = d[p_name]
                    else:
                        continue
                else:
                    val = d.get("value") or d.get("value_raw") or d.get("result") or d.get("test_value") or ""

                if _is_demographic_key(p_name):
                    continue

                ref = d.get("normal_range") or d.get("reference_range_raw") or d.get("reference_range") or d.get("range") or d.get("ref_range") or ""
                unit = d.get("unit") or d.get("unit_raw") or d.get("units") or ""
                flag = d.get("flag")

                try:
                    num_val = float(str(val).strip())
                except (ValueError, TypeError):
                    m_num = re.search(r"[-+]?\d*\.?\d+", str(val))
                    num_val = float(m_num.group(0)) if m_num else None

                records.append({
                    "item_id": self._next_id(),
                    "category": "investigation",
                    "data": {
                        "parameter": p_name,
                        "value_raw": str(val).strip(),
                        "value_numeric": num_val,
                        "unit_raw": str(unit).strip(),
                        "reference_range_raw": str(ref).strip(),
                        "flag": flag,
                    },
                    "assertion_status": "present",
                    "source": {"document_id": document_id, "page": default_page, "evidence": str(entry.get("source", {}).get("evidence") or f"{p_name}: {val}")},
                    "extraction_status": "extracted",
                })
            return records

        if category == "clinical":
            findings_list = []
            if isinstance(parsed, dict):
                findings_list = (
                    parsed.get("clinical_findings")
                    or parsed.get("findings")
                    or parsed.get("items")
                    or []
                )
            elif isinstance(parsed, list):
                findings_list = parsed

            for f in findings_list:
                if not isinstance(f, dict):
                    continue
                d = f.get("data") if isinstance(f.get("data"), dict) else f
                finding_text = d.get("finding") or d.get("diagnosis") or d.get("impression") or d.get("symptom") or d.get("note") or ""
                if not finding_text or _is_demographic_key(finding_text) or len(str(finding_text).strip()) < 2:
                    continue
                records.append({
                    "item_id": self._next_id(),
                    "category": "clinical",
                    "data": {"finding": str(finding_text).strip()},
                    "assertion_status": d.get("status", "present"),
                    "source": {"document_id": document_id, "page": default_page, "evidence": str(finding_text)},
                    "extraction_status": "extracted",
                })
            return records

        raw_items = parsed.get("items", []) if isinstance(parsed, dict) else parsed
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if isinstance(raw_item, dict) and raw_item:
                    records.append(validate_extracted_item(raw_item, document_id, self._next_id()))
        return records

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
                "Extract all items belonging to this agent's scope from these pages.\n"
                f"Additional audit context (if any): {retry_context or 'None'}\n"
                "Return the JSON object specified by the task prompt."
            )
            response = self.client.complete(prompt, user_text, page_chunk)
            parsed = _json_from_text(response)
            page_recs = self._parse_category_items(parsed, category, document_id, page_numbers[0])
            records.extend(page_recs)
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
                "analyzed_parameters": [],
                "potential_conditions": [],
            }

        deterministic_abnormals: list[dict[str, Any]] = []
        deterministic_normals: list[dict[str, Any]] = []

        test_items: list[dict[str, Any]] = []
        seen_params: set[str] = set()

        for item in items:
            cat = str(item.get("category") or "").lower()
            if any(p in cat for p in ("patient", "provider", "facility", "metadata", "demographic")):
                continue
            data = item.get("data", {})
            param = str(data.get("parameter") or data.get("name") or data.get("test") or "").strip()
            if not param or _is_demographic_key(param):
                continue

            p_norm = re.sub(r"[^a-z0-9]", "", param.lower())
            if p_norm in seen_params:
                continue
            seen_params.add(p_norm)
            test_items.append(item)

            ref_str = str(data.get("reference_range_raw") or data.get("reference_range") or data.get("normal_range") or "").strip()
            val_raw = data.get("value_raw") or data.get("value_numeric")
            if val_raw is None:
                continue

            try:
                num_val = float(str(val_raw).strip())
            except (ValueError, TypeError):
                m_num = re.search(r"[-+]?\d*\.?\d+", str(val_raw))
                num_val = float(m_num.group(0)) if m_num else None

            if num_val is None:
                continue

            low, high = _parse_range_bounds(ref_str)
            if low is None and high is None:
                continue

            status, urgency = _evaluate_numeric_status(num_val, low, high)
            unit_str = data.get("unit_raw") or data.get("unit") or ""
            val_display = f"{val_raw} {unit_str}".strip()

            if status == "normal":
                deterministic_normals.append({
                    "item_id": item.get("item_id", ""),
                    "parameter": param,
                    "value": val_display,
                    "reference_range": ref_str,
                    "status": "normal",
                    "urgency": "normal",
                    "note": f"Within normal reference interval ({ref_str}).",
                })
            else:
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

        try:
            response = self.client.complete(
                _system_prompt("interpret.txt"),
                "Interpret the following extracted medical report data. Provide entity-by-entity breakdowns and suggest potential non-definitive health conditions or patterns to discuss with a physician.\n"
                + json.dumps(test_items, ensure_ascii=False),
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

        final_abnormals = [a for a in deterministic_abnormals if not _is_demographic_key(a["parameter"])]
        final_normals = [n for n in deterministic_normals if not _is_demographic_key(n["parameter"])]
        result["abnormal_values"] = final_abnormals
        result["normal_values"] = final_normals

        raw_params = result.get("parameters") or result.get("analyzed_parameters") or []
        model_analyzed = {
            re.sub(r"[^a-z0-9]", "", str(ap.get("parameter", "")).lower()): ap.get("about", "")
            for ap in raw_params if isinstance(ap, dict) and ap.get("about")
        }

        final_analyzed_params: list[dict[str, Any]] = []

        for item in test_items:
            data = item.get("data", {})
            param = str(data.get("parameter") or data.get("name") or data.get("test") or "").strip()
            p_norm = re.sub(r"[^a-z0-9]", "", param.lower())

            val_raw = data.get("value_raw") or data.get("value_numeric")
            unit_str = data.get("unit_raw") or data.get("unit") or ""
            val_display = f"{val_raw} {unit_str}".strip() if val_raw is not None else "Not reported"
            ref_str = str(data.get("reference_range_raw") or data.get("reference_range") or data.get("normal_range") or "").strip() or "Standard reference interval"

            matching_ab = next((a for a in final_abnormals if re.sub(r"[^a-z0-9]", "", a["parameter"].lower()) == p_norm), None)
            matching_norm = next((n for n in final_normals if re.sub(r"[^a-z0-9]", "", n["parameter"].lower()) == p_norm), None)

            if matching_ab:
                st = matching_ab.get("status", "high")
                if "critical" in st:
                    classification = "critical_high" if "high" in st else "critical_low"
                else:
                    classification = "high" if "high" in st else "low"
                clin_sig = matching_ab.get("significance", f"Value is {classification.replace('_', ' ')} relative to reference range.")
            elif matching_norm:
                classification = "in_range"
                clin_sig = "Result is within standard physiological reference interval."
            else:
                classification = "in_range"
                clin_sig = "Parameter observed in report."

            about_text = (
                _get_medical_test_about(param)
                or model_analyzed.get(p_norm)
                or f"{param} is a clinical diagnostic marker measured to evaluate organ health and metabolic balance."
            )

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

        pot_conds = result.get("potential_conditions") or result.get("possible_conditions") or []
        valid_conds: list[dict[str, Any]] = []
        for pc in pot_conds:
            if isinstance(pc, dict) and pc.get("condition"):
                assoc = [f for f in pc.get("associated_findings", []) if not _is_demographic_key(str(f))]
                pc["associated_findings"] = assoc
                pc.setdefault("caution_note", "This is a potential consideration for review with your doctor, NOT a definitive diagnosis.")
                valid_conds.append(pc)

        if not valid_conds and final_abnormals:
            ab_names = [a["parameter"] for a in final_abnormals]
            ab_lowers = {a.lower() for a in ab_names}
            if "chloride" in ab_lowers and ("bicarbonate" in ab_lowers or "co2" in ab_lowers):
                valid_conds.append({
                    "condition": "Potential Acid-Base or Fluid Balance Consideration",
                    "associated_findings": [a for a in ab_names if a.lower() in ("chloride", "bicarbonate", "co2")],
                    "explanation": "Elevated chloride accompanied by lower bicarbonate levels may suggest metabolic acid-base variation or dehydration requiring physician evaluation.",
                    "caution_note": "This is a potential consideration for review with your doctor, NOT a definitive diagnosis.",
                })
            if "creatinine" in ab_lowers or "urea" in ab_lowers:
                valid_conds.append({
                    "condition": "Renal Excretory Function Evaluation",
                    "associated_findings": [a for a in ab_names if a.lower() in ("creatinine", "urea", "b.urea")],
                    "explanation": "Altered kidney filtration markers suggest discussing renal function and hydration status with your doctor.",
                    "caution_note": "This is a potential consideration for review with your doctor, NOT a definitive diagnosis.",
                })
            if "uric acid" in ab_lowers:
                valid_conds.append({
                    "condition": "Hyperuricemia Consideration",
                    "associated_findings": ["Uric Acid"],
                    "explanation": "Elevated uric acid levels may warrant discussing dietary purine intake and evaluating risks for gout or kidney stones with a physician.",
                    "caution_note": "This is a potential consideration for review with your doctor, NOT a definitive diagnosis.",
                })
            if not valid_conds:
                valid_conds.append({
                    "condition": f"Clinical Review for {', '.join(ab_names[:3])}",
                    "associated_findings": ab_names,
                    "explanation": f"The documented measurements for {', '.join(ab_names)} fall outside normal reference limits and warrant physician discussion.",
                    "caution_note": "This is a potential consideration for review with your doctor, NOT a definitive diagnosis.",
                })

        result["potential_conditions"] = valid_conds

        if not result["report_summary"]:
            if final_abnormals:
                ab_names = ", ".join(a["parameter"] for a in final_abnormals[:4])
                result["report_summary"] = f"Diagnostic evaluation identified out-of-range parameters for {ab_names}. All other reported tests were within standard reference limits."
            else:
                result["report_summary"] = "All evaluated laboratory parameters fall within standard physiological reference intervals."

        if not result["patient_alert"]:
            if final_abnormals:
                ab_names = ", ".join(a["parameter"] for a in final_abnormals[:3])
                result["patient_alert"] = f"Your lab results showed out-of-range values for {ab_names}. Please share these findings with your doctor so they can review them in the context of your overall health and medical history."
            else:
                result["patient_alert"] = "Your test results are within normal reference limits. Review these results with your doctor at your next scheduled visit."

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
            # Crucial safeguard: Always run patient and investigations on all pages so tests are never dropped by mapper errors
            if category in {"patient", "investigations"}:
                for page in pages:
                    if page not in selected_pages:
                        selected_pages.append(page)
            else:
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
