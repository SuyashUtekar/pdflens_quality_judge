from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import settings
from .evidence import docx_evidence, pdf_evidence, text_match
from .ingestion import RunInput
from .llm import LLMError, LiteLLMJudge

CONVERSION = json.loads((Path(__file__).parent / "criteria_conversion.json").read_text())
METADATA = json.loads((Path(__file__).parent / "criteria_metadata.json").read_text())
_NUMBERS = re.compile(r"(?<!\w)[+-]?(?:(?:\d{1,4}[-/.]){1,2}\d{1,4}|\d[\d,]*(?:\.\d+)?\s*%?)")
_OUTCOMES = {"PASS", "FAIL", "N/A", "INCONCLUSIVE", "NOT_REQUESTED"}
_SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

METADATA_LLM_IDS = {
    "H01", "H02", "H04", "H05", "F01", "F02", "F03", "F04", "F05", "L03",
    "I01", "I02", "I03", "I04", "I05", "T01", "T02", "T03", "T04", "T05", "T06",
}


def _score(findings: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item["outcome"] for item in findings)
    passed, failed = counts["PASS"], counts["FAIL"]
    denominator = passed + failed
    return {"pass_count": passed, "fail_count": failed, "not_applicable_count": counts["N/A"],
            "not_requested_count": counts["NOT_REQUESTED"], "inconclusive_count": counts["INCONCLUSIVE"],
            "scorable_count": denominator,
            "pass_percentage": round(100 * passed / denominator, 2) if denominator else None,
            "signed_score": round(100 * (passed - failed) / denominator, 2) if denominator else None,
            "raw_score": passed - failed}


def _image_decodes(raw: bytes) -> bool:
    try:
        import pymupdf
        pixmap = pymupdf.Pixmap(raw)
        return pixmap.width > 0 and pixmap.height > 0
    except Exception:
        return False


def _finding(module: str, criterion_id: str, outcome: str, explanation: str, *,
             page_number: int | None = None, severity: str | None = None,
             source_refs: list[str] | None = None, output_refs: list[str] | None = None,
             confidence: float | None = None, human_review: bool | None = None,
             scope: str | None = None) -> dict[str, Any]:
    criterion = next(item for item in (CONVERSION if module == "conversion" else METADATA)["criteria"]
                     if item["id"] == criterion_id)
    if human_review is None:
        human_review = outcome in {"INCONCLUSIVE", "FAIL"}
    return {"finding_id": "", "module": module, "scope": scope or ("page" if page_number else "document"),
            "page_number": page_number, "criterion_id": criterion_id, "category": criterion["category"],
            "criterion": criterion["name"], "outcome": outcome,
            "severity": severity if outcome == "FAIL" else None, "explanation": explanation,
            "source_evidence_refs": source_refs or [], "output_evidence_refs": output_refs or [],
            "confidence": confidence, "human_review_required": bool(human_review)}


def _similarity_candidates(source_page: dict[str, Any], docx_pages: list[dict[str, Any]]) -> dict[str, Any]:
    source_text = source_page["text"]
    candidates: list[dict[str, Any]] = []
    for start in range(len(docx_pages)):
        for width in range(1, min(3, len(docx_pages) - start) + 1):
            span = docx_pages[start:start + width]
            target = " ".join(page["text"] for page in span)
            recall, precision = text_match(source_text, target)
            confidence = 0.75 * recall + 0.25 * precision
            candidates.append({"pages": span, "text": target, "recall": recall,
                               "precision": precision, "confidence": confidence})
    return max(candidates, key=lambda item: (item["confidence"], item["recall"]), default={
        "pages": [], "text": "", "recall": 0.0, "precision": 0.0, "confidence": 0.0})


def _conversion_source_text(page: dict[str, Any], repeated: set[str]) -> str:
    blocks = []
    for block in page["blocks"]:
        text = block["text"]
        _x0, y0, _x1, y1 = block["bbox"]
        top = y1 < page["height_pt"] * 0.2
        bottom = y0 > page["height_pt"] * 0.8
        if text.casefold() in repeated:
            continue
        if re.fullmatch(r"(?:page\s*)?\d{1,5}(?:\s*/\s*\d{1,5})?", text.strip(), flags=re.I) and (top or bottom):
            continue
        blocks.append(text)
    return "\n".join(blocks)


def _llm_findings(judge: LiteLLMJudge | None, module: str, page_number: int,
                  criterion_ids: list[str], payload: dict[str, Any], images: list[bytes],
                  source_refs: set[str], output_refs: set[str], prompt: str) -> list[dict[str, Any]]:
    if not criterion_ids:
        return []
    if judge is None:
        return [_finding(module, cid, "INCONCLUSIVE", "No LiteLLM endpoint is configured for this judgment.",
                         page_number=page_number, human_review=True) for cid in criterion_ids]
    try:
        response = judge.judge(prompt, payload, images)
    except LLMError as exc:
        return [_finding(module, cid, "INCONCLUSIVE", str(exc).replace("\n", " ")[:400],
                         page_number=page_number, human_review=True) for cid in criterion_ids]
    returned: dict[str, bool] = {}
    findings = []
    for item in response["criteria"]:
        if not isinstance(item, dict):
            continue
        cid = item.get("criterion_id")
        if cid not in criterion_ids or cid in returned:
            continue
        outcome = item.get("outcome")
        if outcome not in _OUTCOMES:
            continue
        src = item.get("source_evidence_refs", [])
        out = item.get("output_evidence_refs", [])
        if not isinstance(src, list) or not isinstance(out, list):
            continue
        if any(ref not in source_refs for ref in src) or any(ref not in output_refs for ref in out):
            continue
        severity = item.get("severity") if item.get("severity") in _SEVERITIES else None
        confidence = item.get("confidence")
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = None
        if outcome in {"PASS", "FAIL"} and (confidence is None or confidence < 0.75):
            outcome, severity = "INCONCLUSIVE", None
        if outcome == "PASS" and (not src or not out):
            outcome, severity = "INCONCLUSIVE", None
        if outcome == "FAIL" and (not src or not out or confidence < 0.85 or severity is None):
            outcome, severity = "INCONCLUSIVE", None
        findings.append(_finding(module, cid, outcome,
                                 str(item.get("explanation") or "The judge returned no explanation.")[:1000],
                                 page_number=page_number, severity=severity, source_refs=src, output_refs=out,
                                 confidence=confidence,
                                 human_review=bool(item.get("human_review_required", outcome in {"FAIL", "INCONCLUSIVE"}))))
        returned[cid] = True
    for cid in criterion_ids:
        if cid not in returned:
            findings.append(_finding(module, cid, "INCONCLUSIVE",
                                     "LLM response omitted the criterion or contained unsupported evidence references.",
                                     page_number=page_number, human_review=True))
    return findings


def _conversion(pdf: dict[str, Any], docx: dict[str, Any],
                judge: LiteLLMJudge | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    docx_pages = docx.get("render_pages", [])
    mapped: list[dict[str, Any]] = []
    headers = _furniture_candidates(pdf, "header")
    footers = _furniture_candidates(pdf, "footer")
    for source_page in pdf["pages"]:
        excluded_lines = {item["text"].casefold() for item in
                          headers.get(source_page["page_number"], []) + footers.get(source_page["page_number"], [])}
        comparison_text = _conversion_source_text(source_page, excluded_lines)
        aligned_source_page = {**source_page, "text": comparison_text}
        candidate = _similarity_candidates(aligned_source_page, docx_pages) if docx_pages else {
            "pages": [], "text": "", "recall": 0, "precision": 0, "confidence": 0}
        target_nums = [item["page_number"] for item in candidate["pages"]]
        mapping = {"source_page": source_page["page_number"], "docx_pages": target_nums,
                   "alignment_type": "UNMATCHED", "confidence": round(candidate["confidence"], 3),
                   "text_recall": round(candidate["recall"], 3), "text_precision": round(candidate["precision"], 3)}
        if target_nums and candidate["confidence"] >= 0.18:
            mapping["alignment_type"] = "ONE_TO_MANY" if len(target_nums) > 1 else "ONE_TO_ONE"
            if any(target_nums[0] in other.get("docx_pages", []) for other in mapped):
                mapping["alignment_type"] = "MANY_TO_ONE"
        mapped.append(mapping)
        page_num = source_page["page_number"]
        source_refs = [block["ref"] for block in source_page["blocks"]]
        output_refs = [f"docx:p{number}" for number in target_nums]
        confident = mapping["confidence"] >= 0.65
        all_ids = [item["id"] for item in CONVERSION["criteria"]]
        page_findings: list[dict[str, Any]] = []
        recall, precision = candidate["recall"], candidate["precision"]
        if not docx["available"]:
            page_findings.append(_finding("conversion", "C01", "INCONCLUSIVE", "PDFLens DOCX is missing.",
                                          page_number=page_num, source_refs=source_refs))
        elif not docx_pages:
            page_findings.append(_finding("conversion", "C01", "INCONCLUSIVE",
                                          "DOCX could not be rendered; fixed-page alignment is unavailable.",
                                          page_number=page_num, source_refs=source_refs,
                                          output_refs=[item["ref"] for item in docx["units"][:12]]))
        elif mapping["alignment_type"] == "UNMATCHED" or not confident:
            page_findings.append(_finding("conversion", "C01", "INCONCLUSIVE",
                                          "PDF-to-DOCX page alignment is too weak for a completeness decision.",
                                          page_number=page_num, source_refs=source_refs,
                                          output_refs=output_refs, confidence=mapping["confidence"]))
        else:
            outcome = "PASS" if recall >= 0.92 else "FAIL" if recall < 0.55 else "INCONCLUSIVE"
            page_findings.append(_finding("conversion", "C01", outcome,
                                          f"Token coverage for the best aligned DOCX page span is {recall:.1%}; thresholds are provisional.",
                                          page_number=page_num, severity="HIGH" if outcome == "FAIL" else None,
                                          source_refs=source_refs, output_refs=output_refs,
                                          confidence=mapping["confidence"], human_review=outcome == "INCONCLUSIVE"))
        if confident and candidate["text"]:
            src_nums = Counter(re.sub(r"[\s,]+", "", item).casefold()
                               for item in _NUMBERS.findall(comparison_text))
            out_nums = Counter(re.sub(r"[\s,]+", "", item).casefold()
                               for item in _NUMBERS.findall(candidate["text"]))
            missing, extra = list((src_nums - out_nums).elements()), list((out_nums - src_nums).elements())
            outcome = "FAIL" if missing or extra else "PASS"
            page_findings.append(_finding("conversion", "C04", outcome,
                                          f"Numeric comparison: source-only={missing[:12]}, DOCX-only={extra[:12]}.",
                                          page_number=page_num, severity="HIGH" if missing else "MEDIUM" if extra else None,
                                          source_refs=source_refs, output_refs=output_refs,
                                          confidence=mapping["confidence"]))
        else:
            page_findings.append(_finding("conversion", "C04", "INCONCLUSIVE",
                                          "No reliable fixed-page text alignment is available for numeric comparison.",
                                          page_number=page_num, source_refs=source_refs, output_refs=output_refs,
                                          human_review=True))
        # No exact duplicate page text is evidence of no duplicate; cross-page continuity needs alignment context.
        page_findings.append(_finding("conversion", "C07", "INCONCLUSIVE",
                                      "Cross-page continuity requires adjacent-page mapping and is not inferred from text counts alone.",
                                      page_number=page_num, source_refs=source_refs, output_refs=output_refs,
                                      human_review=True))
        deterministic = {item["criterion_id"] for item in page_findings}
        pending = [cid for cid in all_ids if cid not in deterministic]
        if pending and mapping["alignment_type"] != "UNMATCHED" and docx_pages and confident:
            page_images = [source_page["jpeg"], *[item["jpeg"] for item in candidate["pages"][:3]]]
            payload = {"source_page_number": page_num, "alignment": mapping,
                       "applicable_criteria": [item for item in CONVERSION["criteria"] if item["id"] in pending],
                       "source_evidence": {"blocks": source_page["blocks"], "text": comparison_text[:12000]},
                       "output_evidence": [{"ref": f"docx:p{item['page_number']}", "text": item["text"][:12000]}
                                           for item in candidate["pages"][:3]],
                       "evidence_note": "Page images follow: source PDF page first, then the listed aligned DOCX pages."}
            page_findings.extend(_llm_findings(judge, "conversion", page_num, pending, payload,
                                               page_images, set(source_refs) | {f"pdf:p{page_num}"}, set(output_refs),
                                               "conversion_judge_v1.txt"))
        else:
            for cid in pending:
                page_findings.append(_finding("conversion", cid, "INCONCLUSIVE",
                                              "Reliable rendered DOCX alignment is required for this page-level check.",
                                              page_number=page_num, source_refs=source_refs, output_refs=output_refs,
                                              human_review=True))
        findings.extend(page_findings)
    return findings, {"page_alignment": mapped, "docx_renderer": docx.get("renderer"),
                      "docx_rendered": docx.get("rendered", False),
                      "docx_render_error": docx.get("render_error")}


def _metadata_elements(page: dict[str, Any], field: str) -> list[dict[str, Any]]:
    elements = page.get("elements", []) if isinstance(page, dict) else []
    if not isinstance(elements, list):
        return []
    return [item for item in elements if isinstance(item, dict)
            and str(item.get("type", "")).casefold() == field]


def _furniture_candidates(pdf: dict[str, Any], band: str) -> dict[int, list[dict[str, Any]]]:
    positions: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for page in pdf["pages"]:
        for block in page["blocks"]:
            _x0, y0, _x1, y1 = block["bbox"]
            is_band = y1 < page["height_pt"] * 0.16 if band == "header" else y0 > page["height_pt"] * 0.84
            if is_band:
                key = re.sub(r"\W+", " ", block["text"].casefold()).strip()
                if len(key) > 2:
                    positions.setdefault(key, []).append((page["page_number"], block))
    return {page_number: [block for _num, block in entries]
            for entries in positions.values() if len({num for num, _ in entries}) >= 2
            for page_number, block in entries}


def _metadata(bundle: RunInput, pdf: dict[str, Any], config: dict[str, Any],
              judge: LiteLLMJudge | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    metadata = bundle.metadata or {}
    pages = metadata.get("pages", []) if isinstance(metadata, dict) else []
    pages = pages if isinstance(pages, list) else []
    page_map = {item.get("page_number"): item for item in pages if isinstance(item, dict)}
    page_indices = {item.get("page_number"): index for index, item in enumerate(pages) if isinstance(item, dict)}
    processing = metadata.get("processing", {}) if isinstance(metadata.get("processing"), dict) else {}
    requested_value = config.get("requested_fields", config.get(
        "extraction_fields", processing.get("requested_fields")))
    requested = {str(value).casefold() for value in requested_value} if isinstance(requested_value, list) else None
    schema_errors: list[str] = []
    if bundle.metadata_parse_error:
        schema_errors.append(bundle.metadata_parse_error)
    elif metadata:
        try:
            import jsonschema
        except Exception as exc:
            schema_errors.append(f"Metadata schema validator unavailable ({type(exc).__name__}).")
        else:
            try:
                schema = json.loads(settings.schema_path.read_text(encoding="utf-8"))
                jsonschema.validate(metadata, schema)
            except jsonschema.ValidationError as exc:
                schema_errors.append(f"{exc.json_path}: {exc.message}")
            except Exception as exc:
                schema_errors.append(f"Metadata schema validation failed ({type(exc).__name__}).")

    if bundle.metadata_parse_error:
        findings.append(_finding("metadata", "J01", "FAIL", bundle.metadata_parse_error,
                                 scope="document", severity="HIGH", confidence=1.0))
    elif bundle.metadata_raw is None:
        findings.append(_finding("metadata", "J01", "INCONCLUSIVE", "metadata.json is absent from the ZIP.",
                                 scope="document", human_review=True))
    else:
        findings.append(_finding("metadata", "J01", "PASS", "metadata.json is valid JSON.",
                                 scope="document", confidence=1.0, human_review=False))
    if schema_errors:
        findings.append(_finding("metadata", "J02", "FAIL", "; ".join(schema_errors[:8]),
                                 scope="document", severity="HIGH", confidence=0.99))
        findings.append(_finding("metadata", "J03", "FAIL", "Required schema fields or types are invalid.",
                                 scope="document", severity="HIGH", confidence=0.99))
    elif metadata:
        findings.append(_finding("metadata", "J02", "PASS", "Metadata validates against the bundled canonical v1 schema.",
                                 scope="document", confidence=1.0, human_review=False))
        findings.append(_finding("metadata", "J03", "PASS", "Schema-required metadata fields are present.",
                                 scope="document", confidence=1.0, human_review=False))
    else:
        findings.extend([_finding("metadata", cid, "INCONCLUSIVE", "Metadata JSON is unavailable for validation.",
                                   scope="document") for cid in ("J02", "J03")])

    element_ids: list[str] = []
    referenced_assets: list[str] = []
    for page in pages:
        for element in page.get("elements", []) if isinstance(page, dict) and isinstance(page.get("elements"), list) else []:
            if isinstance(element, dict):
                if element.get("element_id"):
                    element_ids.append(str(element["element_id"]))
                if element.get("artifact_path"):
                    referenced_assets.append(str(element["artifact_path"]))
    duplicates = [item for item, count in Counter(element_ids).items() if count > 1]
    findings.append(_finding("metadata", "J04", "INCONCLUSIVE" if not metadata else "FAIL" if duplicates else "PASS",
                             f"Duplicate element IDs: {duplicates[:12]}" if duplicates else "Element IDs are unique.",
                             scope="document", severity="HIGH" if duplicates else None,
                             confidence=1.0, human_review=bool(duplicates)))
    page_numbers = [item.get("page_number") for item in pages if isinstance(item, dict)]
    duplicate_page_numbers = [number for number, count in Counter(page_numbers).items() if count > 1]
    invalid_pages = [item.get("page_number") if isinstance(item, dict) else None for item in pages
                     if not isinstance(item, dict) or not isinstance(item.get("page_number"), int)
                     or not 1 <= item.get("page_number", 0) <= pdf["page_count"]]
    invalid_pages.extend(duplicate_page_numbers)
    findings.append(_finding("metadata", "J05", "INCONCLUSIVE" if not metadata else "FAIL" if invalid_pages else "PASS",
                             f"Invalid page references: {invalid_pages[:12]}" if invalid_pages
                             else "Page references are within source PDF bounds.", scope="document",
                             severity="HIGH" if invalid_pages else None, confidence=1.0, human_review=bool(invalid_pages)))
    asset_names = set(bundle.assets)
    declared_assets = metadata.get("artifacts", {}) if isinstance(metadata.get("artifacts"), dict) else {}
    referenced_assets.extend(value for key in ("image_paths", "table_snapshot_paths")
                             for value in declared_assets.get(key, []) if isinstance(value, str))
    missing_assets = sorted({name for name in referenced_assets if name not in asset_names})
    findings.append(_finding("metadata", "J06", "INCONCLUSIVE" if not metadata else "FAIL" if missing_assets else "PASS",
                             f"Missing asset references: {missing_assets[:12]}" if missing_assets
                             else "All declared asset references resolve inside the PDFLens ZIP.",
                             scope="document", severity="HIGH" if missing_assets else None,
                             confidence=1.0, human_review=bool(missing_assets)))
    doc_page_count = metadata.get("document", {}).get("page_count") if isinstance(metadata.get("document"), dict) else None
    cross_consistent = doc_page_count == pdf["page_count"] and len(page_map) == len(pages)
    findings.append(_finding("metadata", "J07", "INCONCLUSIVE" if not metadata else "PASS" if cross_consistent else "FAIL",
                             "Metadata document/page counts agree with the source PDF." if cross_consistent
                             else "Document page_count, page records, or unique page references conflict.",
                             scope="document", severity="HIGH" if not cross_consistent else None,
                             confidence=0.99, human_review=not cross_consistent))

    repeated_headers = _furniture_candidates(pdf, "header")
    repeated_footers = _furniture_candidates(pdf, "footer")
    requested_map = {
        "M": "body_text", "H": "header", "F": "footnote", "L": "hyperlink", "I": "image", "T": "table"
    }
    for source_page in pdf["pages"]:
        page_num = source_page["page_number"]
        target = page_map.get(page_num)
        page_findings: list[dict[str, Any]] = []
        page_ref = f"pdf:p{page_num}"
        output_ref = f"/pages/{page_indices.get(page_num, page_num - 1)}"
        if target is None:
            page_findings.append(_finding("metadata", "M02", "FAIL", "Metadata page record is missing.",
                                          page_number=page_num, source_refs=[page_ref],
                                          output_refs=["metadata:/pages"], severity="HIGH", confidence=1.0))
            page_findings.append(_finding("metadata", "M01", "FAIL", "No metadata page record maps to this PDF page.",
                                          page_number=page_num, source_refs=[page_ref], output_refs=["metadata:/pages"],
                                          severity="HIGH", confidence=1.0))
            page_findings.append(_finding("metadata", "M05", "FAIL", "No page-level element structure is available for this source page.",
                                          page_number=page_num, source_refs=[page_ref], output_refs=["metadata:/pages"],
                                          severity="MEDIUM", confidence=1.0))
            groups = {
                "body_text": ("M03", "M04"), "header": ("H01", "H02", "H03"),
                "footer": ("H04", "H05", "H06"), "footnote": ("F01", "F02", "F03", "F04", "F05"),
                "hyperlink": ("L01", "L02", "L03", "L04", "L05"),
                "image": tuple(f"I{i:02d}" for i in range(1, 9)),
                "table": tuple(f"T{i:02d}" for i in range(1, 9)),
            }
            assigned = {"M01", "M02", "M05"}
            for field, ids in groups.items():
                state = "INCONCLUSIVE" if requested is None or field in requested else "NOT_REQUESTED"
                for cid in ids:
                    if cid not in assigned:
                        page_findings.append(_finding("metadata", cid, state,
                                                      f"Page record is absent; {field} evaluation cannot inspect page-level metadata.",
                                                      page_number=page_num, source_refs=[page_ref],
                                                      output_refs=["/pages"],
                                                      human_review=state == "INCONCLUSIVE"))
                        assigned.add(cid)
            findings.extend(page_findings)
            continue
        else:
            page_findings.append(_finding("metadata", "M01", "PASS", "A metadata page record exists for this source page.",
                                          page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                          confidence=1.0, human_review=False))
            if target.get("page_number") == page_num:
                page_findings.append(_finding("metadata", "M02", "PASS", "Metadata page number matches the PDF page.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                              confidence=1.0, human_review=False))
            else:
                page_findings.append(_finding("metadata", "M02", "FAIL", "Metadata record is nested under a different page number.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                              severity="HIGH", confidence=1.0))
        output_text = re.sub(r"\[\[(?:IMAGE|TABLE):[^\]]+\]\]", " ",
                             str(target.get("body_text", "")), flags=re.IGNORECASE)
        repeated_furniture = {block["text"].casefold() for block in
                              repeated_headers.get(page_num, []) + repeated_footers.get(page_num, [])}
        typed_furniture = [(str(element.get("type", "")).casefold(),
                            str(element.get("text", "")).casefold())
                           for element in elements if isinstance(element, dict)]
        expected_blocks = []
        for block in source_page["blocks"]:
            x0, y0, x1, y1 = block["bbox"]
            normalized_block = block["text"].casefold()
            top_band = y1 < source_page["height_pt"] * 0.2
            bottom_band = y0 > source_page["height_pt"] * 0.8
            is_page_number = bool(re.fullmatch(r"(?:page\s*)?\d{1,5}(?:\s*/\s*\d{1,5})?", normalized_block.strip()))
            matching_type = next((kind for kind, text in typed_furniture if text == normalized_block), None)
            if normalized_block in repeated_furniture:
                continue
            if matching_type in {"header", "page_number"} and (top_band or bottom_band):
                continue
            if matching_type in {"footer", "footnote"} and bottom_band:
                continue
            if is_page_number and (top_band or bottom_band):
                continue
            if any(table.get("bbox") and x0 < table["bbox"][2] and x1 > table["bbox"][0]
                   and y0 < table["bbox"][3] and y1 > table["bbox"][1]
                   for table in source_page["tables"]):
                continue
            expected_blocks.append(block["text"])
        expected_body = " ".join(expected_blocks)
        recall, precision = text_match(expected_body, output_text)
        if requested is None:
            for cid in ("M03", "M04"):
                page_findings.append(_finding("metadata", cid, "INCONCLUSIVE",
                                              "Extraction configuration is unavailable; body_text applicability is unknown.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[f"{output_ref}/body_text"],
                                              human_review=True))
        elif "body_text" in requested:
            outcome = "INCONCLUSIVE" if not source_page["text"].strip() else "N/A" if not expected_body.strip() and not output_text.strip() else (
                "PASS" if recall >= 0.92 else "FAIL" if recall < 0.55 else "INCONCLUSIVE")
            page_findings.append(_finding("metadata", "M03", outcome,
                                          f"PDF-to-body_text token coverage is {recall:.1%}; heuristic thresholds are provisional.",
                                          page_number=page_num, source_refs=[item["ref"] for item in source_page["blocks"][:30]],
                                          output_refs=[f"{output_ref}/body_text"],
                                          severity="HIGH" if outcome == "FAIL" else None,
                                          confidence=0.8 if source_page["text"] else 0.1,
                                          human_review=outcome in {"FAIL", "INCONCLUSIVE"}))
            accuracy = "INCONCLUSIVE" if not source_page["text"].strip() else "N/A" if not expected_body.strip() and not output_text.strip() else (
                "PASS" if precision >= 0.9 and recall >= 0.9 else
                "FAIL" if precision < 0.5 and recall > 0.5 else "INCONCLUSIVE")
            page_findings.append(_finding("metadata", "M04", accuracy,
                                          f"Normalized text comparison: source recall={recall:.1%}, body-text precision={precision:.1%}.",
                                          page_number=page_num, source_refs=[page_ref], output_refs=[f"{output_ref}/body_text"],
                                          severity="MEDIUM" if accuracy == "FAIL" else None,
                                          confidence=0.8 if source_page["text"] else 0.1,
                                          human_review=accuracy in {"FAIL", "INCONCLUSIVE"}))
        else:
            for cid in ("M03", "M04"):
                page_findings.append(_finding("metadata", cid, "NOT_REQUESTED", "body_text was not requested in extraction configuration.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                              confidence=1.0, human_review=False))
        elements = target.get("elements", []) if isinstance(target.get("elements"), list) else []
        structure_ok = all(isinstance(item, dict) and isinstance(item.get("type"), str)
                           for item in elements)
        page_findings.append(_finding("metadata", "M05", "PASS" if structure_ok else "FAIL",
                                      "Page elements are grouped as typed records." if structure_ok
                                      else "Page contains an element with missing or invalid type structure.",
                                      page_number=page_num, source_refs=[page_ref], output_refs=[f"{output_ref}/elements"],
                                      severity="MEDIUM" if not structure_ok else None, confidence=1.0,
                                      human_review=not structure_ok))

        for prefix, field, candidates in (("H", "header", repeated_headers), ("H", "footer", repeated_footers)):
            if field == "footer":
                ids = ("H04", "H05", "H06")
            else:
                ids = ("H01", "H02", "H03")
            expected = candidates.get(page_num, [])
            extracted = _metadata_elements(target, field)
            if requested is None:
                for cid in ids:
                    page_findings.append(_finding("metadata", cid, "INCONCLUSIVE",
                                                  f"Extraction configuration is unavailable; {field} applicability is unknown.",
                                                  page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                                  human_review=True))
            elif field not in requested:
                for cid in ids:
                    page_findings.append(_finding("metadata", cid, "NOT_REQUESTED", f"{field} was not requested.",
                                                  page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                                  confidence=1.0, human_review=False))
            elif expected:
                best = max((text_match(item["text"], str(meta.get("text", "")))[0]
                            for item in expected for meta in extracted), default=0.0)
                detected = "PASS" if extracted and best >= 0.8 else "FAIL" if not extracted else "INCONCLUSIVE"
                page_findings.append(_finding("metadata", ids[0], detected,
                                              f"Repeated PDF {field} candidates={len(expected)}; matching metadata records={len(extracted)}.",
                                              page_number=page_num, source_refs=[item["ref"] for item in expected],
                                              output_refs=[f"{output_ref}/elements"],
                                              severity="MEDIUM" if detected == "FAIL" else None,
                                              confidence=0.75 if extracted else 0.9,
                                              human_review=detected in {"FAIL", "INCONCLUSIVE"}))
                page_findings.append(_finding("metadata", ids[1], "PASS" if best >= 0.8 else "FAIL" if extracted else "INCONCLUSIVE",
                                              f"Best normalized {field} text overlap is {best:.1%}.", page_number=page_num,
                                              source_refs=[item["ref"] for item in expected],
                                              output_refs=[f"{output_ref}/elements"],
                                              severity="MEDIUM" if extracted and best < 0.5 else None,
                                              confidence=best if extracted else 0.3,
                                              human_review=not extracted or best < 0.8))
                page_findings.append(_finding("metadata", ids[2], "PASS" if best >= 0.8 else "FAIL" if extracted else "INCONCLUSIVE",
                                              f"{len(extracted)} {field} element(s) are nested on the corresponding page record.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[f"{output_ref}/elements"],
                                              severity="HIGH" if extracted and best < 0.5 else None,
                                              confidence=best if extracted else 0.2,
                                              human_review=not extracted or best < 0.8))
            else:
                status = "INCONCLUSIVE"
                for cid in ids:
                    page_findings.append(_finding("metadata", cid, status,
                                                  f"No repeated source {field} candidate was established from text-layer evidence.",
                                                  page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                                  human_review=status == "INCONCLUSIVE"))

        # Deterministic link checks; association and visual text checks remain separate.
        if requested is None:
            for cid in ("L01", "L02", "L03", "L04", "L05"):
                page_findings.append(_finding("metadata", cid, "INCONCLUSIVE",
                                              "Extraction configuration is unavailable; hyperlink applicability is unknown.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                              human_review=True))
        elif "hyperlink" not in requested:
            for cid in ("L01", "L02", "L03", "L04", "L05"):
                page_findings.append(_finding("metadata", cid, "NOT_REQUESTED", "hyperlink was not requested.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                              confidence=1.0, human_review=False))
        else:
            links = source_page["links"]
            extracted_links = _metadata_elements(target, "hyperlink")
            output_targets = {str(item.get("target_uri") or item.get("attributes", {}).get("uri", ""))
                              for item in extracted_links}
            if not links and not extracted_links:
                outcome = "N/A"
            elif not links and extracted_links:
                outcome = "FAIL"
            elif links and not extracted_links:
                outcome = "FAIL"
            else:
                outcome = "PASS" if all(link["uri"] in output_targets for link in links) else "FAIL"
            page_findings.append(_finding("metadata", "L01", outcome,
                                          f"PDF links={len(links)}; metadata hyperlink elements={len(extracted_links)}.",
                                          page_number=page_num, source_refs=[item["ref"] for item in links] or [page_ref],
                                          output_refs=[f"{output_ref}/elements"],
                                          severity="MEDIUM" if outcome == "FAIL" else None,
                                          confidence=0.9 if outcome != "N/A" else None,
                                          human_review=outcome in {"FAIL", "INCONCLUSIVE"}))
            missing = [link["uri"] for link in links if link["uri"] not in output_targets]
            link_state = ("N/A" if not links and not extracted_links else
                          "FAIL" if missing or (extracted_links and not links) else "PASS")
            page_findings.append(_finding("metadata", "L02", link_state,
                                          f"Targets missing from metadata: {missing[:8]}" if missing else "Extracted link targets match PDF links.",
                                          page_number=page_num, source_refs=[item["ref"] for item in links] or [page_ref],
                                          output_refs=[f"{output_ref}/elements"],
                                          severity="HIGH" if missing else None, confidence=0.9, human_review=bool(missing)))
            for cid, explanation in (("L03", "Visible-text to target association needs visual judgment."),
                                     ("L04", "Links nested on page records inherit this page association."),
                                     ("L05", "URI syntax can be checked locally; external reachability is not evaluated.")):
                if not links and not extracted_links:
                    status, conf = "N/A", None
                elif cid == "L04":
                    status, conf = "PASS", 1.0
                elif cid == "L05":
                    bad = [item for item in extracted_links if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:",
                                                                            str(item.get("target_uri") or ""))]
                    status, conf = ("FAIL" if bad else "PASS"), 1.0
                else:
                    status, conf = "INCONCLUSIVE", None
                page_findings.append(_finding("metadata", cid, status, explanation, page_number=page_num,
                                              source_refs=[item["ref"] for item in links],
                                              output_refs=[f"{output_ref}/elements"], confidence=conf,
                                              severity="LOW" if cid == "L05" and status == "FAIL" else None,
                                              human_review=status in {"FAIL", "INCONCLUSIVE"}))

        # Exact asset path checks can be done per page; scope/count and visual match may be uncertain.
        for prefix, field, source_count in (("I", "image", source_page["image_count"]),
                                            ("T", "table", len(source_page["tables"]))):
            if requested is None:
                ids = [f"{prefix}{number:02d}" for number in range(1, 9)]
                for cid in ids:
                    page_findings.append(_finding("metadata", cid, "INCONCLUSIVE",
                                                  f"Extraction configuration is unavailable; {field} applicability is unknown.",
                                                  page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                                  human_review=True))
                continue
            if field not in requested:
                ids = [f"{prefix}{number:02d}" for number in range(1, 9)]
                for cid in ids:
                    page_findings.append(_finding("metadata", cid, "NOT_REQUESTED", f"{field} was not requested.",
                                                  page_number=page_num, source_refs=[page_ref], output_refs=[output_ref],
                                                  confidence=1.0, human_review=False))
                continue
            extracted = _metadata_elements(target, field)
            if prefix == "I":
                refs = ["I01", "I02", "I03", "I04", "I05", "I06", "I07", "I08"]
                no_in_scope_image = source_count == 0 and not extracted
                if no_in_scope_image:
                    status, reason = "N/A", "No PDF image xref or image metadata record is present."
                elif source_count is None:
                    status, reason = "INCONCLUSIVE", "PDF image detection unavailable."
                elif source_count == 0 and not extracted:
                    status, reason = "N/A", "No embedded PDF image or image metadata record is present."
                elif source_count > 0 and extracted:
                    status, reason = "INCONCLUSIVE", "PDF image xrefs may include decorative or repeated assets; semantic in-scope matching is pending."
                else:
                    status, reason = "INCONCLUSIVE", "Image scope and visual correspondence are not confirmed by image count alone."
                page_findings.append(_finding("metadata", "I01", status, reason, page_number=page_num,
                                              source_refs=[page_ref], output_refs=[f"{output_ref}/elements"],
                                              human_review=status == "INCONCLUSIVE"))
                page_findings.append(_finding("metadata", "I02", status, f"PDF image xref count={source_count}; metadata image count={len(extracted)}.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[f"{output_ref}/elements"],
                                              human_review=status == "INCONCLUSIVE"))
                for cid in refs[2:5] + ["I08"]:
                    page_findings.append(_finding("metadata", cid, "N/A" if no_in_scope_image else "INCONCLUSIVE",
                                                  "No in-scope image exists." if no_in_scope_image else "Image position, description, and in-scope duplicate status require visual/contract evidence.",
                                                  page_number=page_num, source_refs=[page_ref], output_refs=[f"{output_ref}/elements"],
                                                  human_review=not no_in_scope_image))
                missing_image_assets = [item.get("artifact_path") or "<missing artifact_path>" for item in extracted
                                         if not item.get("artifact_path") or item.get("artifact_path") not in asset_names]
                image_asset_state = "N/A" if not extracted else "FAIL" if missing_image_assets else "PASS"
                page_findings.append(_finding("metadata", "I06", image_asset_state,
                                              f"Unresolved image artifact paths: {missing_image_assets[:6]}" if missing_image_assets else "Image artifact references resolve.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[f"{output_ref}/elements"],
                                              severity="HIGH" if missing_image_assets else None, confidence=1.0,
                                              human_review=bool(missing_image_assets)))
                image_assets = [bundle.assets[item["artifact_path"]] for item in extracted
                                if item.get("artifact_path") in bundle.assets]
                image_integrity = [_image_decodes(asset) for asset in image_assets]
                valid = (None if not image_assets else all(image_integrity))
                page_findings.append(_finding("metadata", "I07", "N/A" if no_in_scope_image else "INCONCLUSIVE" if valid is None else "PASS" if valid else "FAIL",
                                              "All referenced image assets decoded successfully." if valid else "At least one image asset could not be decoded." if valid is False else "No resolvable image asset was available for integrity check.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[f"{output_ref}/elements"],
                                              severity="HIGH" if valid is False else None,
                                              confidence=1.0 if valid is not None else None,
                                              human_review=valid is None or valid is False))
            else:
                if not source_page["table_detection_available"]:
                    table_status = "INCONCLUSIVE"
                    table_reason = "PDF table detection was unavailable."
                elif source_count == 0 and not extracted:
                    table_status = "N/A"
                    table_reason = "No PDF table detected and no table record exists."
                elif source_count == len(extracted):
                    table_status = "PASS"
                    table_reason = "Detected PDF and metadata table counts agree (structure remains separately reviewable)."
                else:
                    table_status = "FAIL"
                    table_reason = f"PDF table candidates={source_count}; metadata table records={len(extracted)}."
                page_findings.append(_finding("metadata", "T01", table_status, table_reason,
                                              page_number=page_num, source_refs=[item["ref"] for item in source_page["tables"]] or [page_ref],
                                              output_refs=[f"{output_ref}/elements"],
                                              severity="HIGH" if table_status == "FAIL" else None,
                                              confidence=0.8 if source_page["table_detection_available"] else None,
                                              human_review=table_status in {"FAIL", "INCONCLUSIVE"}))
                page_findings.append(_finding("metadata", "T02", table_status,
                                              f"PDF table candidates={source_count}; metadata table records={len(extracted)}.",
                                              page_number=page_num, source_refs=[item["ref"] for item in source_page["tables"]] or [page_ref],
                                              output_refs=[f"{output_ref}/elements"],
                                              severity="HIGH" if table_status == "FAIL" else None,
                                              confidence=0.8 if source_page["table_detection_available"] else None,
                                              human_review=table_status in {"FAIL", "INCONCLUSIVE"}))
                for cid in ("T03", "T04", "T05", "T06", "T08"):
                    page_findings.append(_finding("metadata", cid, "INCONCLUSIVE",
                                                  "Table page/structure/cell/location assessment requires reliable region matching.",
                                                  page_number=page_num, source_refs=[item["ref"] for item in source_page["tables"]],
                                                  output_refs=[f"{output_ref}/elements"], human_review=True))
                snapshots = [item.get("artifact_path") for item in extracted if item.get("artifact_path")]
                bad_snapshots = [path for path in snapshots if path not in asset_names]
                snapshot_state = ("FAIL" if bad_snapshots else "PASS" if snapshots else
                                  "INCONCLUSIVE" if extracted else "N/A")
                page_findings.append(_finding("metadata", "T07", snapshot_state,
                                              f"Unresolved table snapshot paths: {bad_snapshots[:6]}" if bad_snapshots else "Table snapshot references resolve." if snapshots else "Table record exists without a snapshot reference, but snapshot requirement is not explicit." if extracted else "No table record is present.",
                                              page_number=page_num, source_refs=[page_ref], output_refs=[f"{output_ref}/elements"],
                                              severity="HIGH" if bad_snapshots else None, confidence=1.0,
                                              human_review=snapshot_state in {"FAIL", "INCONCLUSIVE"}))

        # Footnotes and any remaining contextual decisions are explicit, non-scoring uncertainty for now.
        footnotes = _metadata_elements(target, "footnote")
        if requested is None:
            foot_status, foot_explanation = "INCONCLUSIVE", "Extraction configuration is unavailable; footnote applicability is unknown."
        elif "footnote" not in requested:
            foot_status, foot_explanation = "NOT_REQUESTED", "footnote was not requested."
        else:
            foot_status, foot_explanation = "INCONCLUSIVE", "Footnote marker detection and text-region association are not yet reliable deterministic checks."
        for cid in ("F01", "F02", "F03", "F04", "F05"):
            page_findings.append(_finding("metadata", cid, foot_status,
                                          f"{foot_explanation} Metadata records={len(footnotes)}.",
                                          page_number=page_num, source_refs=[page_ref], output_refs=[f"{output_ref}/elements"],
                                          human_review=foot_status == "INCONCLUSIVE"))

        replace_with_llm = [item for item in page_findings
                            if item["criterion_id"] in METADATA_LLM_IDS
                            and item["outcome"] == "INCONCLUSIVE"
                            and (requested is not None and requested_map.get(item["criterion_id"][:1]) in requested)]
        llm_pending = [item["criterion_id"] for item in replace_with_llm]
        page_findings = [item for item in page_findings if item not in replace_with_llm]
        if llm_pending:
            payload = {"page_number": page_num, "applicable_criteria": [item for item in METADATA["criteria"] if item["id"] in llm_pending],
                       "source_pdf_text_blocks": source_page["blocks"],
                       "metadata_page_record": target,
                       "extraction_configuration": sorted(requested) if requested is not None else None,
                       "coordinate_system": target.get("coordinate_system"),
                       "pdf_link_evidence": source_page["links"],
                       "note": "Image attached is the original PDF page. Metadata element paths are /pages/{zero-based index}/elements/{zero-based index}."}
            asset_images = []
            for element_index, element in enumerate(elements):
                asset_path = element.get("artifact_path") if isinstance(element, dict) else None
                if asset_path in bundle.assets and len(asset_images) < 4:
                    raw_asset = bundle.assets[asset_path]
                    if raw_asset.startswith(b"\x89PNG\r\n\x1a\n") or raw_asset.startswith(b"\xff\xd8"):
                        payload.setdefault("asset_evidence", []).append({
                            "metadata_json_path": f"/pages/{page_num - 1}/elements/{element_index}/artifact_path",
                            "zip_member": asset_path,
                        })
                        asset_images.append(raw_asset)
            metadata_output_refs = {output_ref, f"{output_ref}/elements", f"{output_ref}/body_text"}
            for element_index, element in enumerate(elements):
                metadata_output_refs.add(f"{output_ref}/elements/{element_index}")
                if isinstance(element, dict):
                    metadata_output_refs.update(f"{output_ref}/elements/{element_index}/{key}"
                                                for key in element)
                    attributes = element.get("attributes")
                    if isinstance(attributes, dict):
                        metadata_output_refs.update(
                            f"{output_ref}/elements/{element_index}/attributes/{key}" for key in attributes)
            metadata_output_refs.update(str(item.get("artifact_path")) for item in elements
                                        if isinstance(item, dict) and item.get("artifact_path"))
            page_findings.extend(_llm_findings(judge, "metadata", page_num, llm_pending, payload,
                                               [source_page["jpeg"], *asset_images],
                                               {block["ref"] for block in source_page["blocks"]}
                                               | {page_ref} | {item["ref"] for item in source_page["links"]}
                                               | {item["ref"] for item in source_page["tables"]},
                                               metadata_output_refs, "metadata_judge_v1.txt"))
        findings.extend(page_findings)
    # Emit document-level findings for any missing metadata page that was not otherwise handled.
    return findings, {"metadata_present": bundle.metadata_raw is not None, "metadata_schema_errors": schema_errors,
                      "requested_fields": sorted(requested) if requested is not None else None,
                      "source_page_count": pdf["page_count"], "metadata_page_count": len(pages)}


def evaluate(bundle: RunInput, modules: list[str], config: dict[str, Any]) -> dict[str, Any]:
    pdf = pdf_evidence(bundle.pdf_bytes)
    docx = docx_evidence(bundle.docx_bytes, render=settings.render_docx)
    try:
        judge: LiteLLMJudge | None = LiteLLMJudge()
        llm_state = {"configured": True, "provider": "LiteLLM OpenAI-compatible proxy", "model": judge.model}
    except LLMError as exc:
        judge = None
        llm_state = {"configured": False, "provider": "LiteLLM OpenAI-compatible proxy", "model": None,
                     "note": str(exc)}
    selected = {module.casefold() for module in modules}
    findings: list[dict[str, Any]] = []
    conversion_info: dict[str, Any] = {}
    metadata_info: dict[str, Any] = {}
    if "conversion" in selected:
        conversion_findings, conversion_info = _conversion(pdf, docx, judge)
        findings.extend(conversion_findings)
    if "metadata" in selected:
        metadata_findings, metadata_info = _metadata(bundle, pdf, config, judge)
        findings.extend(metadata_findings)
    for index, finding in enumerate(findings, start=1):
        finding["finding_id"] = f"finding-{index:07d}"
    module_reports = {}
    for module in selected:
        values = [item for item in findings if item["module"] == module]
        score = _score(values)
        threshold = float(config.get(f"{module}_pass_candidate_at", 95))
        review_below = float(config.get(f"{module}_review_required_below", 85))
        critical = sum(item["severity"] == "CRITICAL" and item["outcome"] == "FAIL" for item in values)
        percentage = score["pass_percentage"]
        gate = ("NOT_EVALUATED" if score["scorable_count"] == 0 else "FAIL"
                if critical or percentage < review_below else "REVIEW_REQUIRED"
                if percentage < threshold or score["inconclusive_count"] else "PASS")
        module_reports[module] = {**score, "quality_gate": gate, "critical_failures": critical,
                                  "high_failures": sum(item["severity"] == "HIGH" and item["outcome"] == "FAIL" for item in values),
                                  "finding_count": len(values),
                                  "pages_with_failures": len({item["page_number"] for item in values
                                                               if item["outcome"] == "FAIL" and item["page_number"]})}
    gates = [item["quality_gate"] for item in module_reports.values()]
    overall = ("NOT_EVALUATED" if not gates else "FAIL" if "FAIL" in gates else
               "REVIEW_REQUIRED" if "REVIEW_REQUIRED" in gates or "NOT_EVALUATED" in gates else "PASS")
    return {"evaluation_id": f"pdflens-eval-{uuid.uuid4()}",
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "COMPLETED",
            "overall_quality_gate": overall, "selected_modules": sorted(selected),
            "module_reports": module_reports,
            "document_id": (config.get("document_id") or (bundle.metadata or {}).get("document", {}).get("source_filename")),
            "criteria_versions": {"conversion": CONVERSION["criteria_version"], "metadata": METADATA["criteria_version"]},
            "judge_application_version": __version__, "llm": llm_state,
            "prompt_versions": {"conversion": "conversion_judge_v1", "metadata": "metadata_judge_v1"},
            "source_pdf": {"sha256": hashlib.sha256(bundle.pdf_bytes).hexdigest(),
                           "file_name": bundle.pdf_name, "page_count": pdf["page_count"],
                           "metadata": pdf["metadata"]},
            "pdflens_package": {"sha256": bundle.package_sha256, "members": bundle.members,
                                "docx_member": bundle.docx_name, "assets_count": len(bundle.assets)},
            "conversion": {**conversion_info, "docx_text_available": bool(docx.get("text")),
                           "docx_text_chars": len(docx.get("text", "")),
                           "docx_unit_count": len(docx.get("units", [])),
                           "docx_table_row_count": len(docx.get("table_rows", [])),
                           "docx_image_count": docx.get("image_count", 0)},
            "metadata": metadata_info, "findings": findings,
            "evidence_notes": [
                "The original PDF is the reference source; DOCX is evaluated as a separate conversion output.",
                "Conversion page mapping uses text similarity over contiguous rendered DOCX page spans and does not assume page N equals page N.",
                "LibreOffice rendering is optional. Without it, fixed-page alignment and visual conversion checks remain INCONCLUSIVE.",
                "Metadata requested-field applicability is taken from the explicit evaluation config or metadata.processing.requested_fields.",
                "Initial criteria and thresholds are proposed and require calibration on human-reviewed examples.",
            ]}
