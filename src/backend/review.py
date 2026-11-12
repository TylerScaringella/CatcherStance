from __future__ import annotations

import html
import io
import json
import time
import zipfile
from pathlib import Path

from .storage import RunLocation, atomic_write_json, run_lock

FINAL_STANCES = ("KneeDown - Left", "KneeDown - Right", "Squat")
_CANONICAL_TO_EXPORT = {
    "LKD": "KneeDown - Left",
    "Knee-down Left": "KneeDown - Left",
    "RKD": "KneeDown - Right",
    "Knee-down Right": "KneeDown - Right",
    "Squat": "Squat",
}


def review_path(location: RunLocation) -> Path:
    return location.path / "review_overrides.json"


def load_review_state(location: RunLocation) -> dict:
    try:
        payload = json.loads(review_path(location).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        payload = {}
    overrides = payload.get("overrides") if isinstance(payload, dict) else None
    return {"version": 1, "overrides": overrides if isinstance(overrides, dict) else {}}


def save_review(location: RunLocation, clip_id: str, stance: str, note: str = "") -> dict:
    if stance not in FINAL_STANCES:
        raise ValueError("stance must be KneeDown - Left, KneeDown - Right, or Squat")
    with run_lock(location.run_id):
        state = load_review_state(location)
        history = state["overrides"].setdefault(clip_id, [])
        entry = {
            "final_stance": stance,
            "note": str(note or "")[:1000],
            "reviewed_at": time.time(),
        }
        history.append(entry)
        atomic_write_json(review_path(location), state)
    return entry


def review_for_clip(state: dict, clip_id: str) -> dict:
    history = state.get("overrides", {}).get(clip_id, [])
    latest = history[-1] if history else None
    return {
        "review_status": "reviewed" if latest else "unreviewed",
        "final_stance": latest.get("final_stance") if latest else None,
        "review_note": latest.get("note", "") if latest else "",
        "reviewed_at": latest.get("reviewed_at") if latest else None,
        "review_history": history,
    }


def enrich_result(result: dict, state: dict) -> dict:
    enriched = dict(result)
    original = result.get("stance") or None
    review = review_for_clip(state, str(result.get("clip_id") or ""))
    enriched.update({
        "model_stance": original,
        "final_stance": review["final_stance"] or original,
        **review,
    })
    return enriched


def review_summary(results: list[dict]) -> dict:
    reviewed = sum(row.get("review_status") == "reviewed" for row in results)
    return {
        "total": len(results),
        "reviewed": reviewed,
        "unresolved": len(results) - reviewed,
        "ready": bool(results) and reviewed == len(results),
    }


def export_value(row: dict) -> str | None:
    stance = row.get("final_stance")
    if stance in FINAL_STANCES:
        return stance
    return _CANONICAL_TO_EXPORT.get(str(stance))


def _cell(value: str | int, reference: str, style: str = "") -> str:
    if isinstance(value, int):
        return f'<c r="{reference}"{style}><v>{value}</v></c>'
    safe = html.escape(str(value), quote=True)
    return f'<c r="{reference}" t="inlineStr"{style}><is><t>{safe}</t></is></c>'


def build_xlsx(results: list[dict]) -> bytes:
    rows = sorted(results, key=lambda row: int(row.get("pitch_index") or 0))
    sheet_rows = [
        '<row r="1">'
        + _cell("Pitch number", "A1", ' s="1"')
        + _cell("Catcher Pose", "B1", ' s="1"')
        + "</row>"
    ]
    for index, row in enumerate(rows, start=2):
        pitch_number = int(row.get("pitch_index") or index - 1)
        pose = export_value(row)
        sheet_rows.append(
            f'<row r="{index}">{_cell(pitch_number, f"A{index}")}{_cell(pose or "", f"B{index}")}</row>'
        )
    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<cols><col min="1" max="1" width="16" customWidth="1"/><col min="2" max="2" width="24" customWidth="1"/></cols>'
        f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Catcher Review" sheetId="1" r:id="rId1"/></sheets></workbook>')
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellXfs count="2"><xf/><xf fontId="1" applyFont="1"/></cellXfs></styleSheet>')
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    workbook_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in {
            "[Content_Types].xml": content_types,
            "_rels/.rels": root_rels,
            "xl/workbook.xml": workbook,
            "xl/_rels/workbook.xml.rels": workbook_rels,
            "xl/worksheets/sheet1.xml": sheet,
            "xl/styles.xml": styles,
        }.items():
            archive.writestr(name, content)
    return output.getvalue()
