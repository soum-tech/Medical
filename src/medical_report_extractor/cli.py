from __future__ import annotations

import argparse
import sys

from .pipeline import ExtractionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured information from a medical PDF or image using a local vision-language model.")
    parser.add_argument("--input", required=True, help="Path to a medical PDF or image (PNG/JPG/WEBP).")
    parser.add_argument("--output-dir", default="output", help="Directory for JSON outputs (default: output).")
    args = parser.parse_args()

    try:
        pipeline = ExtractionPipeline()
        report_path, audit_path = pipeline.run(args.input, args.output_dir)
        dashboard_path = report_path.parent / "dashboard.html"
        print(f"Extraction complete.\nReport:    {report_path}\nAudit:     {audit_path}\nDashboard: {dashboard_path}")
    except Exception as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
