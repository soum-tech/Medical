# Medical Report Extractor

A lightweight, staged medical-document information extraction pipeline that runs a local vision-language model directly through Hugging Face Transformers.

The project focuses on **faithful extraction**, not medical diagnosis or treatment advice. It uses document mapping, category-specific extraction calls, reconciliation, and a page-wise completeness audit with targeted retries. Prompts remain editable text files outside the Python code.

## Scope

**Included**
- PDF and image ingestion.
- Page-wise document and section mapping.
- Specialized extraction for patient/provider details, clinical information, investigations/results, medications, and care/procedures/follow-up.
- Source-page and evidence preservation.
- Reconciliation that flags possible duplicates and conflicts without silently deleting records.
- Completeness review and configurable targeted re-extraction.
- Pydantic validation and local JSON output.
- Direct local model loading through Transformers. No vLLM server or API endpoint is required.

**Not included in this version**
- Clinical reasoning, differential diagnosis, treatment recommendations, or autonomous triage.
- Knowledge-graph construction or database storage.
- Benchmarking and test suites.

## Architecture

```text
PDF / image
    |
    v
Document ingestion (text + page images)
    |
    v
Document mapper (local model call per page)
    |
    v
Specialized extraction calls
    |-- Patient / provider
    |-- Clinical information
    |-- Investigations and results
    |-- Medications
    |-- Procedures, care plan and follow-up
    |
    v
Reconciliation (duplicate/conflict suggestions; records are preserved)
    |
    v
Page-wise completeness audit
    |
    +---- targeted re-extraction for identified omissions (limited retries)
    |
    v
Validation and JSON output
```

The model and processor are loaded once when the extraction pipeline starts and reused across all calls. The mapper routes pages to relevant extraction categories. If a page cannot be mapped reliably, the pipeline runs all extraction categories on that page rather than silently skipping it. The verifier compares each original page with the records attributed to that page. A verifier finding is treated as a candidate omission until a targeted extraction call checks the source.

## Repository layout

```text
medical-report-extractor/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── prompts/
│   ├── shared.txt
│   ├── map_document.txt
│   ├── extract_patient.txt
│   ├── extract_clinical.txt
│   ├── extract_investigations.txt
│   ├── extract_medications.txt
│   ├── extract_care.txt
│   ├── reconcile.txt
│   └── verify.txt
├── models/                         # Local model weights; not committed to Git
├── input/                          # Put medical reports here
├── output/                         # Generated JSON files
└── src/medical_report_extractor/
    ├── __init__.py
    ├── cli.py
    ├── documents.py
    ├── local_model.py
    ├── pipeline.py
    └── schemas.py
```

## Requirements

- Python 3.10 or newer.
- A local Hugging Face Transformers-format Qwen3.5 checkpoint.
- PyTorch installed for your operating system and hardware. For NVIDIA GPU inference, install a PyTorch build compatible with your CUDA driver/runtime.
- A recent Transformers version with Qwen3.5 support, plus Accelerate and Torchvision.

The code uses the official Qwen3.5 Transformers loading pattern with `AutoProcessor` and `AutoModelForMultimodalLM`. See the [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) for model-specific details.

## Installation

Run these commands from the repository root. Use the same Python environment in which you downloaded the model, if that environment already has a compatible PyTorch installation.

```bash
python -m pip install --upgrade pip

# Install/upgrade the model runtime dependencies.
# If PyTorch is not installed, first install the correct build for your hardware:
# https://pytorch.org/get-started/locally/
python -m pip install --upgrade "transformers>=5.3.0" "accelerate>=1.0.0" "torchvision>=0.21.0"

# Install the extraction project.
python -m pip install -e .
```

If PyTorch is already installed and `torch.cuda.is_available()` is `True`, do not reinstall PyTorch unnecessarily. Check it with:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Configure the local model

Copy `.env.example` to `.env`:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux/macOS
cp .env.example .env
```

Set the local model directory in `.env`:

```env
MODEL_PATH=models/Qwen3.5-0.8B
MODEL_DEVICE_MAP=auto
MODEL_LOCAL_FILES_ONLY=true
MODEL_MAX_NEW_TOKENS=4096

RENDER_DPI=144
MAX_PAGES_PER_CALL=1
MAX_TARGETED_RETRIES=1
```

`MODEL_PATH` can be an absolute path or a path relative to the repository root. For example, if the model was downloaded with:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    local_dir="models/Qwen3.5-0.8B",
)
```

then `MODEL_PATH=models/Qwen3.5-0.8B` is the expected setting.

`MODEL_LOCAL_FILES_ONLY=true` prevents Transformers from trying to fetch model files from Hugging Face. Set it to `false` only when you intentionally want to permit missing files to be downloaded.

`MODEL_DEVICE_MAP=auto` lets Accelerate place the model on available hardware. It can also fall back to CPU, but CPU inference may be slow. The model is loaded once and remains in memory for the pipeline run.

## Run

Place the report in `input/`, then run from the repository root:

```bash
python -m medical_report_extractor.cli \
  --input input/medical_report.pdf \
  --output-dir output
```

Windows PowerShell:

```powershell
python -m medical_report_extractor.cli `
  --input "input/medical_report.pdf" `
  --output-dir "output"
```

Supported input formats: PDF, PNG, JPG/JPEG, and WEBP.

The output directory contains:

```text
output/
├── medical_report.json
└── extraction_audit.json
```

- `medical_report.json`: document map, extracted records, reconciliation observations, and pipeline metadata.
- `extraction_audit.json`: page-level mapping, verifier findings, retry outcomes, and unresolved items.

## Extraction record format

Records use a common envelope while the `data` object is category-specific:

```json
{
  "item_id": "ITEM_0001",
  "category": "investigations",
  "data": {
    "parameter": "Hemoglobin",
    "value_raw": "9.2",
    "value_numeric": 9.2,
    "unit_raw": "g/dL",
    "reference_range_raw": "12-16"
  },
  "assertion_status": "present",
  "source": {
    "document_id": "medical_report",
    "page": 1,
    "evidence": "Hemoglobin: 9.2 g/dL",
    "region": "middle table"
  },
  "extraction_status": "extracted"
}
```

The pipeline preserves original textual values and units. Numeric parsing is optional and does not replace the raw value. Missing, ambiguous, illegible, negative, and uncertain statements must remain distinguishable. Conflicting or repeated findings are retained with their separate source references.

## Customize prompts

Edit files in `prompts/`. `shared.txt` is prepended to each task prompt. The task prompts define the scope and expected JSON response for each call. Keep output keys and category labels aligned with the schemas and routing logic in `src/medical_report_extractor/` when changing prompt contracts.

## Privacy and safety

- Inference runs locally in the Python process. The code does not send report text or page images to a remote model API.
- Model loading is configured for local files only by default. Transformers may still use normal local runtime/cache mechanisms.
- The application writes outputs locally and does not intentionally log document contents.
- Do not commit medical reports, generated JSON, `.env` files, model weights, or other patient-identifiable data. The included `.gitignore` excludes common local input/output locations and `models/`.
- Extraction is not a clinical decision. Verify important values against the original document and use qualified clinical review for medical interpretation.
- The pipeline does not guarantee complete or error-free extraction. Illegible or ambiguous content is surfaced for review rather than guessed.

## Current limitations

- The whole rendered page is sent to the VLM; lengthy reports require many sequential model calls.
- The mapper and verifier depend on the selected model's visual and textual understanding.
- Table-heavy, low-resolution, handwritten, rotated, or multi-column documents may require a higher render resolution or manual review.
- Targeted retries are bounded by `MAX_TARGETED_RETRIES` to avoid unbounded loops.
- This version produces local JSON files; it does not store patient records in a database.
