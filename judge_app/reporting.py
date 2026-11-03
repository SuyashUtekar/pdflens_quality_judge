from __future__ import annotations

import json
from io import BytesIO
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

_HEADER = PatternFill("solid", fgColor="1F4E78")
_OUTCOME = {"PASS": "E2F0D9", "FAIL": "FCE4D6", "INCONCLUSIVE": "FFF2CC",
            "N/A": "E7E6E6", "NOT_REQUESTED": "DDEBF7"}


def _safe(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        return "'" + value
    return value


def _sheet(workbook: Workbook, name: str, headers: list[str], rows: list[list[Any]]) -> None:
    sheet = workbook.create_sheet(name)
    sheet.append(headers)
    for cell in sheet[1]:
        cell.fill = _HEADER
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in rows:
        sheet.append([_safe(item) for item in row])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for index, header in enumerate(headers, start=1):
        width = min(64, max(14, len(header) + 2,
                            max((len(str(sheet.cell(r, index).value or ""))
                                 for r in range(2, min(sheet.max_row, 40) + 1)), default=14) + 2))
        sheet.column_dimensions[get_column_letter(index)].width = width
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for row in range(2, sheet.max_row + 1):
        outcome_col = headers.index("outcome") + 1 if "outcome" in headers else None
        if outcome_col:
            color = _OUTCOME.get(sheet.cell(row, outcome_col).value)
            if color:
                sheet.cell(row, outcome_col).fill = PatternFill("solid", fgColor=color)


def build_xlsx(report: dict[str, Any]) -> bytes:
    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    summary_rows = [
        ["Evaluation ID", report["evaluation_id"]], ["Status", report["status"]],
        ["Overall quality gate", report["overall_quality_gate"]],
        ["Document", report.get("document_id")], ["Selected modules", report["selected_modules"]],
        ["PDF file", report["source_pdf"]["file_name"]], ["PDF pages", report["source_pdf"]["page_count"]],
        ["PDFLens ZIP SHA-256", report["pdflens_package"]["sha256"]],
        ["LLM configured", report["llm"]["configured"]], ["LLM provider", report["llm"]["provider"]],
        ["LLM model", report["llm"].get("model")],
    ]
    for module, result in report["module_reports"].items():
        summary_rows.extend([[f"{module} quality gate", result["quality_gate"]],
                             [f"{module} pass percentage", result["pass_percentage"]],
                             [f"{module} scorable criteria", result["scorable_count"]],
                             [f"{module} PASS", result["pass_count"]], [f"{module} FAIL", result["fail_count"]],
                             [f"{module} INCONCLUSIVE", result["inconclusive_count"]],
                             [f"{module} N/A", result["not_applicable_count"]],
                             [f"{module} NOT_REQUESTED", result["not_requested_count"]]])
    summary.append(["Field", "Value"])
    for cell in summary[1]:
        cell.fill = _HEADER
        cell.font = Font(bold=True, color="FFFFFF")
    for row in summary_rows:
        summary.append([_safe(value) for value in row])
    summary.column_dimensions["A"].width = 34
    summary.column_dimensions["B"].width = 84
    _sheet(workbook, "Findings", ["finding_id", "module", "scope", "page_number", "criterion_id",
                                  "category", "criterion", "outcome", "severity", "explanation",
                                  "confidence", "human_review_required", "source_evidence_refs",
                                  "output_evidence_refs"],
           [[item.get(key) for key in ("finding_id", "module", "scope", "page_number", "criterion_id",
                                       "category", "criterion", "outcome", "severity", "explanation",
                                       "confidence", "human_review_required", "source_evidence_refs",
                                       "output_evidence_refs")] for item in report["findings"]])
    for module in report["selected_modules"]:
        rows = [item for item in report["findings"] if item["module"] == module]
        _sheet(workbook, f"{module.title()} Findings", ["finding_id", "scope", "page_number", "criterion_id",
               "category", "criterion", "outcome", "severity", "explanation", "confidence",
               "human_review_required", "source_evidence_refs", "output_evidence_refs"],
               [[item.get(key) for key in ("finding_id", "scope", "page_number", "criterion_id", "category",
                                          "criterion", "outcome", "severity", "explanation", "confidence",
                                          "human_review_required", "source_evidence_refs", "output_evidence_refs")]
                for item in rows])
    if "conversion" in report["selected_modules"]:
        _sheet(workbook, "Page Alignment", ["source_pdf_page", "docx_page_span", "alignment_type",
               "confidence", "text_recall", "text_precision"],
               [[item.get("source_page"), item.get("docx_pages"), item.get("alignment_type"),
                 item.get("confidence"), item.get("text_recall"), item.get("text_precision")]
                for item in report.get("conversion", {}).get("page_alignment", [])])
    _sheet(workbook, "Audit", ["Field", "Value"], [
        ["criteria_versions", report["criteria_versions"]], ["source_pdf", report["source_pdf"]],
        ["pdflens_package", report["pdflens_package"]], ["conversion", report["conversion"]],
        ["metadata", report["metadata"]], ["prompt_versions", report["prompt_versions"]],
        ["evaluation_configuration", report.get("configuration", {})],
        ["evidence_notes", report["evidence_notes"]],
    ])
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def report_json_bytes(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
