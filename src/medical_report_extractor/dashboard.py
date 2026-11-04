from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_dashboard_html(
    report_data: dict[str, Any],
    audit_data: dict[str, Any],
    page_images: list[str],
    output_path: Path,
) -> None:
    """Generate a self-contained interactive HTML dashboard for medical report review."""

    report_json_str = json.dumps(report_data, ensure_ascii=False)
    audit_json_str = json.dumps(audit_data, ensure_ascii=False)

    model_name = report_data.get("pipeline_metadata", {}).get("model", "Local Model")
    source_name = report_data.get("source_file_name", "Input Document")
    items = report_data.get("items", [])
    patient_name = "Patient"
    patient_age = "N/A"
    patient_gender = "N/A"
    ef_value = None
    diagnoses: list[str] = []

    for item in items:
        data = item.get("data", {})
        cat = item.get("category", "")
        if cat == "patient":
            patient_name = data.get("name") or patient_name
            patient_age = data.get("age_raw") or data.get("age_numeric") or patient_age
            patient_gender = data.get("gender") or patient_gender
        elif "ejection fraction" in str(data.get("parameter", "")).lower() or "e.f." in str(data.get("parameter_raw", "")).lower():
            ef_value = data.get("value_raw") or data.get("value_numeric")
        elif "diagnosis" in cat or "clinical" in cat:
            diag = data.get("diagnosis_name") or data.get("finding")
            if diag and diag not in diagnoses:
                diagnoses.append(diag)

    # Render image tags (base64 embedded data URLs)
    if page_images:
        images_html = "".join(
            f'<div style="text-align:center; margin-bottom:20px;">'
            f'<div style="font-size:0.75rem; color:#94a3b8; margin-bottom:6px; font-weight:600;">PAGE {idx + 1} OF {len(page_images)}</div>'
            f'<img src="{img_data}" alt="Page {idx + 1}" style="max-width:100%; height:auto; border-radius:6px; box-shadow:0 10px 25px rgba(0,0,0,0.5); border:1px solid #334155;">'
            f'</div>'
            for idx, img_data in enumerate(page_images)
        )
    else:
        images_html = (
            '<div style="padding:40px; color:#94a3b8; text-align:center;">'
            'No document page image available.'
            '</div>'
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Medical Report Extractor – Clinical Audit & Review Dashboard</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-dim: rgba(56, 189, 248, 0.15);
      --success: #34d399;
      --warning: #fbbf24;
      --danger: #f87171;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
    
    header {{
      background: #090d16;
      border-bottom: 1px solid var(--card-border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }}
    .brand {{ display: flex; align-items: center; gap: 12px; font-weight: 700; font-size: 1.15rem; }}
    .badge {{ padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }}
    .badge-success {{ background: rgba(52, 211, 153, 0.2); color: var(--success); border: 1px solid var(--success); }}
    .badge-primary {{ background: var(--primary-dim); color: var(--primary); border: 1px solid var(--primary); }}

    .workspace {{ display: flex; flex: 1; overflow: hidden; }}
    .pane-doc {{ width: 48%; border-right: 1px solid var(--card-border); display: flex; flex-direction: column; background: #0b1120; }}
    .pane-header {{ padding: 10px 16px; background: #111827; border-bottom: 1px solid var(--card-border); font-size: 0.85rem; font-weight: 600; color: var(--text-muted); display: flex; justify-content: space-between; }}
    .image-container {{ flex: 1; overflow-y: auto; padding: 16px; }}

    .pane-data {{ width: 52%; display: flex; flex-direction: column; overflow: hidden; }}
    .nav-tabs {{ display: flex; background: #111827; border-bottom: 1px solid var(--card-border); padding: 0 16px; flex-wrap: wrap; }}
    .nav-tab {{ padding: 12px 18px; font-size: 0.85rem; font-weight: 600; color: var(--text-muted); cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s; white-space: nowrap; }}
    .nav-tab.active {{ color: var(--primary); border-color: var(--primary); background: var(--primary-dim); }}
    .tab-content {{ flex: 1; overflow-y: auto; padding: 20px; }}

    .filter-bar {{ display: flex; gap: 10px; margin-bottom: 16px; align-items: center; }}
    .search-input {{ flex: 1; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 6px; padding: 8px 14px; color: var(--text); font-size: 0.85rem; }}
    .search-input:focus {{ outline: none; border-color: var(--primary); }}

    .item-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 14px; margin-bottom: 12px; transition: border 0.15s; }}
    .item-card:hover {{ border-color: var(--primary); }}
    .card-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
    .item-id {{ font-family: monospace; font-size: 0.75rem; color: var(--text-muted); }}
    .item-category {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase; padding: 2px 8px; border-radius: 4px; }}
    .cat-investigation {{ background: #1e3a8a; color: #93c5fd; }}
    .cat-diagnosis {{ background: #581c87; color: #d8b4fe; }}
    .cat-patient {{ background: #064e3b; color: #6ee7b7; }}
    .cat-other {{ background: #374151; color: #d1d5db; }}

    .card-title {{ font-size: 1rem; font-weight: 600; margin-bottom: 6px; color: #fff; }}
    .kv-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; font-size: 0.85rem; margin-bottom: 8px; }}
    .kv-item {{ background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; }}
    .kv-label {{ color: var(--text-muted); font-size: 0.75rem; display: block; }}
    .kv-val {{ font-weight: 600; }}

    .evidence-box {{ background: #090d16; border-left: 3px solid var(--primary); padding: 6px 10px; font-size: 0.8rem; color: #94a3b8; font-family: monospace; margin-top: 6px; border-radius: 0 4px 4px 0; }}

    .summary-section {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 20px; margin-bottom: 16px; line-height: 1.6; font-size: 0.95rem; }}
    .summary-section h3 {{ margin-bottom: 12px; color: var(--primary); font-size: 1.1rem; }}
    .summary-section h4 {{ margin: 12px 0 6px 0; color: #e2e8f0; font-size: 0.95rem; }}
    .summary-section ul {{ padding-left: 20px; margin-bottom: 12px; }}
    .summary-section li {{ margin-bottom: 4px; }}

    .conflict-box {{ background: rgba(251, 191, 36, 0.08); border: 1px solid var(--warning); border-radius: 8px; padding: 14px; margin-bottom: 12px; }}
    .conflict-title {{ font-weight: 700; color: var(--warning); font-size: 0.85rem; margin-bottom: 4px; }}

    /* Clinical Interpretation Tab Styles */
    .interp-summary-card {{
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border: 1px solid var(--primary);
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 20px;
    }}
    .interp-summary-card h3 {{
      color: var(--primary);
      font-size: 1.1rem;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .interp-summary-card p {{
      color: #cbd5e1;
      line-height: 1.7;
      font-size: 0.95rem;
    }}

    .urgency-badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .urgency-critical {{ background: rgba(248, 113, 113, 0.2); color: #f87171; border: 1px solid #f87171; }}
    .urgency-significant {{ background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid #fbbf24; }}
    .urgency-mild {{ background: rgba(253, 224, 71, 0.15); color: #fde047; border: 1px solid #a3a32a; }}
    .urgency-normal {{ background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid #34d399; }}

    .abnormal-card {{
      background: var(--card-bg);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 10px;
      border-left: 4px solid;
      transition: transform 0.1s;
    }}
    .abnormal-card:hover {{ transform: translateX(4px); }}
    .abnormal-card.border-critical {{ border-left-color: #f87171; }}
    .abnormal-card.border-significant {{ border-left-color: #fbbf24; }}
    .abnormal-card.border-mild {{ border-left-color: #fde047; }}
    .abnormal-card.border-normal {{ border-left-color: #34d399; }}

    .abnormal-card .param-name {{ font-weight: 700; font-size: 0.95rem; color: #fff; margin-bottom: 4px; }}
    .abnormal-card .param-detail {{ font-size: 0.82rem; color: #94a3b8; margin-bottom: 2px; }}
    .abnormal-card .param-detail strong {{ color: #cbd5e1; }}
    .abnormal-card .significance {{ font-size: 0.85rem; color: #cbd5e1; margin-top: 8px; padding-top: 8px; border-top: 1px solid #334155; line-height: 1.5; }}

    .risk-flag-card {{
      background: rgba(248, 113, 113, 0.06);
      border: 1px solid rgba(248, 113, 113, 0.3);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 10px;
    }}
    .risk-flag-card.risk-significant {{
      background: rgba(251, 191, 36, 0.06);
      border-color: rgba(251, 191, 36, 0.3);
    }}
    .risk-flag-card.risk-mild {{
      background: rgba(253, 224, 71, 0.06);
      border-color: rgba(253, 224, 71, 0.2);
    }}
    .risk-flag-card .flag-title {{ font-weight: 700; font-size: 0.9rem; color: #fff; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }}
    .risk-flag-card .flag-desc {{ font-size: 0.85rem; color: #cbd5e1; line-height: 1.5; }}

    .correlation-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 10px;
    }}
    .correlation-card .corr-title {{ font-weight: 600; font-size: 0.9rem; color: #fff; margin-bottom: 6px; }}
    .correlation-card .corr-sig {{ font-size: 0.85rem; color: #cbd5e1; line-height: 1.5; }}
    .correlation-card .corr-ids {{ font-size: 0.75rem; color: var(--text-muted); font-family: monospace; margin-top: 6px; }}

    .patient-alert-box {{
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(52, 211, 153, 0.08) 100%);
      border: 1px solid var(--primary);
      border-radius: 10px;
      padding: 20px;
      margin-top: 20px;
    }}
    .patient-alert-box h3 {{
      color: var(--primary);
      font-size: 1rem;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .patient-alert-box p {{
      color: #e2e8f0;
      line-height: 1.7;
      font-size: 0.95rem;
    }}

    .section-header {{
      font-size: 0.9rem;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 1px;
      margin: 20px 0 12px 0;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .section-header .count-badge {{
      background: var(--primary-dim);
      color: var(--primary);
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
    }}

    .normal-values-toggle {{
      background: rgba(52, 211, 153, 0.08);
      border: 1px solid rgba(52, 211, 153, 0.2);
      border-radius: 8px;
      padding: 12px 16px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--success);
      transition: background 0.15s;
    }}
    .normal-values-toggle:hover {{ background: rgba(52, 211, 153, 0.15); }}
    .normal-values-list {{ display: none; margin-top: 10px; }}
    .normal-values-list.expanded {{ display: block; }}
    .normal-value-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 12px;
      font-size: 0.82rem;
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }}
    .normal-value-row:last-child {{ border-bottom: none; }}
    .normal-value-row .nv-param {{ color: #cbd5e1; font-weight: 600; }}
    .normal-value-row .nv-val {{ color: var(--success); }}
    .normal-value-row .nv-note {{ color: var(--text-muted); font-size: 0.78rem; max-width: 50%; text-align: right; }}

    /* Entity parameter card & Potential disease styles */
    .analyzed-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 14px;
      transition: border 0.15s, transform 0.1s;
    }}
    .analyzed-card:hover {{ border-color: var(--primary); transform: translateY(-2px); }}
    .analyzed-card.status-high {{ border-left: 4px solid #fbbf24; }}
    .analyzed-card.status-critical {{ border-left: 4px solid #f87171; }}
    .analyzed-card.status-low {{ border-left: 4px solid #60a5fa; }}
    .analyzed-card.status-in_range {{ border-left: 4px solid #34d399; }}

    .param-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }}
    .param-title {{ font-size: 1.05rem; font-weight: 700; color: #fff; }}
    .param-about {{
      font-size: 0.85rem;
      color: #94a3b8;
      line-height: 1.5;
      margin-bottom: 12px;
      background: rgba(255, 255, 255, 0.02);
      padding: 8px 12px;
      border-radius: 6px;
      border: 1px dashed rgba(255, 255, 255, 0.07);
    }}
    .param-metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin-bottom: 10px;
    }}
    .metric-pill {{
      background: rgba(15, 23, 42, 0.6);
      padding: 8px 12px;
      border-radius: 6px;
      border: 1px solid var(--card-border);
    }}
    .metric-label {{ font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; display: block; margin-bottom: 2px; }}
    .metric-value {{ font-size: 0.95rem; font-weight: 700; color: #fff; }}

    .badge-classification {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .class-in_range {{ background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid #34d399; }}
    .class-high {{ background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid #fbbf24; }}
    .class-low {{ background: rgba(96, 165, 250, 0.2); color: #60a5fa; border: 1px solid #60a5fa; }}
    .class-critical {{ background: rgba(248, 113, 113, 0.2); color: #f87171; border: 1px solid #f87171; }}

    .disclaimer-banner {{
      background: rgba(251, 191, 36, 0.08);
      border: 1px solid var(--warning);
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 16px;
      display: flex;
      align-items: flex-start;
      gap: 12px;
      font-size: 0.88rem;
      color: #fde68a;
      line-height: 1.5;
    }}
    .potential-disease-card {{
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
      border: 1px solid rgba(251, 191, 36, 0.4);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 12px;
    }}
    .disease-title {{ font-size: 1rem; font-weight: 700; color: #fbbf24; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }}
    .disease-findings {{ font-size: 0.78rem; color: #94a3b8; margin-bottom: 8px; }}
    .disease-exp {{ font-size: 0.88rem; color: #e2e8f0; line-height: 1.6; margin-bottom: 10px; }}
    .disease-caution {{
      display: inline-block;
      font-size: 0.75rem;
      color: #94a3b8;
      background: rgba(255, 255, 255, 0.04);
      padding: 4px 10px;
      border-radius: 4px;
      border-left: 2px solid var(--warning);
    }}
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
      <span>Medical Report Reviewer</span>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
      <span class="badge badge-primary">Model: {model_name}</span>
      <span class="badge badge-success">Audit: Complete</span>
    </div>
  </header>

  <div class="workspace">
    <div class="pane-doc">
      <div class="pane-header">
        <span>ORIGINAL SOURCE: {source_name}</span>
        <span>EMBEDDED PREVIEW</span>
      </div>
      <div class="image-container">
        {images_html}
      </div>
    </div>

    <div class="pane-data">
      <div class="nav-tabs">
        <div class="nav-tab active" onclick="switchTab('items')">Extracted Records (<span id="count">{len(items)}</span>)</div>
        <div class="nav-tab" onclick="switchTab('clinical')">Physician Clinical Summary</div>
        <div class="nav-tab" onclick="switchTab('patient')">Patient-Friendly Translation</div>
        <div class="nav-tab" onclick="switchTab('reconciliation')">Reconciliation &amp; Audit</div>
        <div class="nav-tab" onclick="switchTab('interpretation')">&#128300; Clinical Interpretation</div>
      </div>

      <!-- Tab 1: Extracted Items -->
      <div id="tab-items" class="tab-content">
        <div class="filter-bar">
          <input type="text" id="searchBox" class="search-input" placeholder="Search parameters, diagnoses, evidence..." oninput="filterItems()">
        </div>
        <div id="itemsContainer"></div>
      </div>

      <!-- Tab 2: Physician SBAR Summary -->
      <div id="tab-clinical" class="tab-content" style="display:none;">
        <div class="summary-section">
          <h3>Physician Handoff / Clinical SBAR Summary</h3>
          <p><strong>Patient:</strong> {patient_name} ({patient_age} / {patient_gender}) | <strong>Report ID:</strong> {report_data.get('document_id', 'N/A')}</p>
          <hr style="border:0; border-top:1px solid var(--card-border); margin:12px 0;">
          <h4>1. Primary Clinical Observations</h4>
          <ul>
            {'<li><strong>Ejection Fraction:</strong> ' + str(ef_value) + '%</li>' if ef_value else ''}
            {''.join(f'<li><strong>Observation:</strong> {d}</li>' for d in diagnoses)}
          </ul>
          <h4>2. Extracted Diagnostic Scope</h4>
          <p>Extracted {len(items)} clinical parameters and observations across all document sections with strict source grounding.</p>
        </div>
      </div>

      <!-- Tab 3: Patient Friendly -->
      <div id="tab-patient" class="tab-content" style="display:none;">
        <div class="summary-section">
          <h3>Understanding Your Test Results</h3>
          <p>This report summarizes key measurements taken during your diagnostic evaluation.</p>
          <h4>Key Findings</h4>
          <ul>
            {'<li><strong>Pumping Function:</strong> Your estimated ejection fraction is ' + str(ef_value) + '%.</li>' if ef_value else ''}
            {''.join(f'<li>{d}</li>' for d in diagnoses)}
          </ul>
          <h4>Next Step</h4>
          <p>Always review these structured findings with your attending doctor or specialist to guide your personalized care plan.</p>
        </div>
      </div>

      <!-- Tab 4: Reconciliation & Audit -->
      <div id="tab-reconciliation" class="tab-content" style="display:none;">
        <div class="summary-section">
          <h3>Completeness Audit</h3>
          <p><strong>Total Items Extracted:</strong> {len(items)}</p>
          <p><strong>Pages Audited:</strong> {len(audit_data.get('page_audits', []))}</p>
        </div>
        <div id="conflictsContainer"></div>
      </div>

      <!-- Tab 5: Clinical Interpretation -->
      <div id="tab-interpretation" class="tab-content" style="display:none;">
        <div id="interpretationContainer"></div>
      </div>
    </div>
  </div>

  <script>
    const reportData = {report_json_str};
    const auditData = {audit_json_str};

    function renderItems(items) {{
      const container = document.getElementById('itemsContainer');
      document.getElementById('count').textContent = items.length;
      container.innerHTML = '';

      items.forEach(item => {{
        const div = document.createElement('div');
        div.className = 'item-card';

        let catClass = 'cat-other';
        const cat = (item.category || '').toLowerCase();
        if (cat.includes('result') || cat.includes('investigation') || cat.includes('finding')) catClass = 'cat-investigation';
        else if (cat.includes('diagnosis') || cat.includes('clinical')) catClass = 'cat-diagnosis';
        else if (cat.includes('patient') || cat.includes('provider')) catClass = 'cat-patient';

        let title = item.data.parameter || item.data.diagnosis_name || item.data.finding || item.data.name || item.data.facility_name || item.category;

        let gridHtml = '';
        for (const [k, v] of Object.entries(item.data || {{}})) {{
          if (v !== null && typeof v !== 'object') {{
            gridHtml += `<div class="kv-item"><span class="kv-label">${{k}}</span><span class="kv-val">${{v}}</span></div>`;
          }}
        }}

        div.innerHTML = `
          <div class="card-top">
            <span class="item-id">${{item.item_id}}</span>
            <span class="item-category ${{catClass}}">${{item.category}}</span>
          </div>
          <div class="card-title">${{title}}</div>
          ${{gridHtml ? `<div class="kv-grid">${{gridHtml}}</div>` : ''}}
          <div class="evidence-box">Evidence: "${{item.source?.evidence || 'N/A'}}"</div>
        `;
        container.appendChild(div);
      }});
    }}

    function renderConflicts() {{
      const container = document.getElementById('conflictsContainer');
      const recon = reportData.reconciliation || {{}};
      const conflicts = recon.possible_conflicts || [];
      const duplicates = recon.possible_duplicates || [];

      let html = '';
      if (conflicts.length > 0) {{
        conflicts.forEach(c => {{
          html += `
            <div class="conflict-box">
              <div class="conflict-title">POSSIBLE SOURCE DISCREPANCY (${{c.item_ids.join(', ')}})</div>
              <p style="font-size:0.85rem; color:#cbd5e1;">${{c.reason}}</p>
            </div>
          `;
        }});
      }}
      if (duplicates.length > 0) {{
        duplicates.forEach(d => {{
          html += `
            <div class="item-card" style="border-left: 3px solid var(--primary);">
              <div style="font-weight:600; font-size:0.85rem; color:var(--primary); margin-bottom:4px;">REPEATED OBSERVATION (${{d.item_ids.join(', ')}})</div>
              <p style="font-size:0.85rem; color:#94a3b8;">${{d.reason}}</p>
            </div>
          `;
        }});
      }}
      if (!html) {{
        html = '<p style="color:var(--text-muted); font-size:0.85rem;">No unresolved conflicts or duplicate anomalies detected.</p>';
      }}
      container.innerHTML = html;
    }}

    function renderInterpretation() {{
      const container = document.getElementById('interpretationContainer');
      const interp = reportData.interpretation;

      if (!interp) {{
        container.innerHTML = '<div class="summary-section"><h3>Clinical Interpretation</h3><p style="color:var(--text-muted);">No clinical interpretation data available. Re-run the pipeline to generate semantic analysis.</p></div>';
        return;
      }}

      let html = '';

      // Report Summary Card
      if (interp.report_summary) {{
        html += `
          <div class="interp-summary-card">
            <h3>&#128203; Clinical Summary &amp; Overview</h3>
            <p>${{interp.report_summary}}</p>
          </div>
        `;
      }}

      // Potential Health Considerations / Diseases to Discuss (Cautious & Non-Definitive)
      const potConds = interp.potential_conditions || interp.possible_conditions || [];
      if (potConds.length > 0) {{
        html += `
          <div class="section-header">&#129658; Potential Health Considerations to Discuss with Your Doctor <span class="count-badge">${{potConds.length}}</span></div>
          <div class="disclaimer-banner">
            <span style="font-size:1.2rem;">&#9888;&#65039;</span>
            <div>
              <strong>Important Cautionary Notice:</strong> The potential conditions listed below are non-definitive patterns and physiological possibilities suggested by your combined test results for review with your healthcare provider. <em>They do NOT constitute a confirmed diagnosis.</em> Only a licensed physician can diagnose medical conditions.
            </div>
          </div>
        `;
        potConds.forEach(pc => {{
          const findings = pc.associated_findings ? pc.associated_findings.join(', ') : '';
          html += `
            <div class="potential-disease-card">
              <div class="disease-title">&#128302; ${{pc.condition || 'Potential Consideration'}}</div>
              ${{findings ? `<div class="disease-findings"><strong>Associated markers:</strong> ${{findings}}</div>` : ''}}
              <div class="disease-exp">${{pc.explanation || ''}}</div>
              <div class="disease-caution">${{pc.caution_note || 'Possibility for physician evaluation, not a confirmed diagnosis.'}}</div>
            </div>
          `;
        }});
      }}

      // Entity-by-Entity Analyzed Parameters Breakdown
      const analyzed = interp.analyzed_parameters || interp.parameters || [];
      if (analyzed.length > 0) {{
        html += `<div class="section-header">&#128202; Test Parameter Breakdown &amp; Classification <span class="count-badge">${{analyzed.length}}</span></div>`;
        analyzed.forEach(ap => {{
          const cls = (ap.classification || 'in_range').toLowerCase();
          let borderCls = 'status-in_range';
          let badgeCls = 'class-in_range';
          let badgeText = 'IN RANGE';

          if (cls.includes('critical') && cls.includes('high')) {{
            borderCls = 'status-critical'; badgeCls = 'class-critical'; badgeText = 'CRITICAL HIGH';
          }} else if (cls.includes('critical') && cls.includes('low')) {{
            borderCls = 'status-critical'; badgeCls = 'class-critical'; badgeText = 'CRITICAL LOW';
          }} else if (cls.includes('high')) {{
            borderCls = 'status-high'; badgeCls = 'class-high'; badgeText = 'HIGH';
          }} else if (cls.includes('low')) {{
            borderCls = 'status-low'; badgeCls = 'class-low'; badgeText = 'LOW';
          }}

          html += `
            <div class="analyzed-card ${{borderCls}}">
              <div class="param-header">
                <span class="param-title">${{ap.parameter || 'Diagnostic Test'}}</span>
                <span class="badge-classification ${{badgeCls}}">${{badgeText}}</span>
              </div>
              ${{ap.about ? `<div class="param-about"><strong>About this test:</strong> ${{ap.about}}</div>` : ''}}
              <div class="param-metrics-grid">
                <div class="metric-pill">
                  <span class="metric-label">Fetched Value</span>
                  <span class="metric-value">${{ap.fetched_value || 'N/A'}}</span>
                </div>
                <div class="metric-pill">
                  <span class="metric-label">Normal Range</span>
                  <span class="metric-value">${{ap.normal_range || 'Not specified'}}</span>
                </div>
                <div class="metric-pill">
                  <span class="metric-label">Status</span>
                  <span class="metric-value">${{badgeText}}</span>
                </div>
              </div>
              ${{ap.clinical_significance ? `<div style="font-size:0.85rem; color:#cbd5e1; margin-top:8px; line-height:1.5;"><strong>Clinical Context:</strong> ${{ap.clinical_significance}}</div>` : ''}}
            </div>
          `;
        }});
      }}

      // Risk Flags Section
      const riskFlags = interp.risk_flags || [];
      if (riskFlags.length > 0) {{
        html += `<div class="section-header">&#128680; Risk Flags <span class="count-badge">${{riskFlags.length}}</span></div>`;
        riskFlags.forEach(rf => {{
          const urgency = (rf.urgency || 'significant').toLowerCase();
          let riskClass = 'risk-significant';
          if (urgency === 'critical') riskClass = '';
          else if (urgency === 'mild') riskClass = 'risk-mild';

          html += `
            <div class="risk-flag-card ${{riskClass}}">
              <div class="flag-title">
                ${{rf.flag || 'Risk Flag'}}
                <span class="urgency-badge urgency-${{urgency}}">${{urgency}}</span>
              </div>
              <div class="flag-desc">${{rf.description || ''}}</div>
              ${{rf.related_item_ids && rf.related_item_ids.length ? `<div style="font-size:0.75rem; color:var(--text-muted); font-family:monospace; margin-top:8px;">Related: ${{rf.related_item_ids.join(', ')}}</div>` : ''}}
            </div>
          `;
        }});
      }}

      // Clinical Correlations Section
      const correlations = interp.clinical_correlations || [];
      if (correlations.length > 0) {{
        html += `<div class="section-header">&#128279; Clinical Correlations <span class="count-badge">${{correlations.length}}</span></div>`;
        correlations.forEach(cc => {{
          const urgency = (cc.urgency || 'mild').toLowerCase();
          html += `
            <div class="correlation-card">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="corr-title">${{cc.finding || 'Correlation'}}</div>
                <span class="urgency-badge urgency-${{urgency}}">${{urgency}}</span>
              </div>
              <div class="corr-sig">${{cc.significance || ''}}</div>
              ${{cc.related_item_ids && cc.related_item_ids.length ? `<div class="corr-ids">Items: ${{cc.related_item_ids.join(', ')}}</div>` : ''}}
            </div>
          `;
        }});
      }}

      // Normal Values Section (collapsed by default)
      const normals = interp.normal_values || [];
      if (normals.length > 0) {{
        html += `
          <div class="section-header">&#9989; Normal Values <span class="count-badge">${{normals.length}}</span></div>
          <div class="normal-values-toggle" onclick="toggleNormals()">
            &#9660; Show ${{normals.length}} values within normal range
          </div>
          <div id="normalValuesList" class="normal-values-list">
        `;
        normals.forEach(nv => {{
          html += `
            <div class="normal-value-row">
              <span class="nv-param">${{nv.parameter || 'Unknown'}}</span>
              <span class="nv-val">${{nv.value || 'N/A'}}</span>
              <span class="nv-note">${{nv.note || ''}}</span>
            </div>
          `;
        }});
        html += '</div>';
      }}

      // Patient Alert Box
      if (interp.patient_alert) {{
        html += `
          <div class="patient-alert-box">
            <h3>&#128172; What This Means For You</h3>
            <p>${{interp.patient_alert}}</p>
          </div>
        `;
      }}

      // Fallback if no content
      if (!html) {{
        html = '<div class="summary-section"><h3>Clinical Interpretation</h3><p style="color:var(--text-muted);">The interpretation stage returned no clinical findings.</p></div>';
      }}

      container.innerHTML = html;
    }}

    function toggleNormals() {{
      const list = document.getElementById('normalValuesList');
      const btn = list.previousElementSibling;
      if (list.classList.contains('expanded')) {{
        list.classList.remove('expanded');
        const count = (reportData.interpretation?.normal_values || []).length;
        btn.innerHTML = '&#9660; Show ' + count + ' values within normal range';
      }} else {{
        list.classList.add('expanded');
        btn.innerHTML = '&#9650; Hide normal values';
      }}
    }}

    function filterItems() {{
      const q = document.getElementById('searchBox').value.toLowerCase();
      const filtered = (reportData.items || []).filter(item => {{
        return JSON.stringify(item).toLowerCase().includes(q);
      }});
      renderItems(filtered);
    }}

    function switchTab(tab) {{
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
      
      const idx = ['items', 'clinical', 'patient', 'reconciliation', 'interpretation'].indexOf(tab);
      document.querySelectorAll('.nav-tab')[idx].classList.add('active');
      document.getElementById(`tab-${{tab}}`).style.display = 'block';
    }}

    renderItems(reportData.items || []);
    renderConflicts();
    renderInterpretation();
  </script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
