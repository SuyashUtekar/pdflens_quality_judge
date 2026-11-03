from __future__ import annotations

import base64
import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .ingestion import RunInput


def clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:[.,:/%-][\w]+)*", text.casefold(), flags=re.UNICODE)


def text_match(left: str, right: str) -> tuple[float, float]:
    from collections import Counter

    a, b = Counter(tokens(left)), Counter(tokens(right))
    if not a or not b:
        return 0.0, 0.0
    overlap = sum((a & b).values())
    recall = overlap / sum(a.values())
    precision = overlap / sum(b.values())
    return recall, precision


def _jpeg(page: Any, scale: float = 1.2) -> bytes:
    import pymupdf
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return pixmap.tobytes("jpeg")


def pdf_evidence(pdf_bytes: bytes) -> dict[str, Any]:
    import pymupdf

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    try:
        for index, page in enumerate(document, start=1):
            blocks = []
            for block_number, block in enumerate(page.get_text("blocks"), start=1):
                if len(block) < 5 or (len(block) > 6 and int(block[6]) != 0):
                    continue
                text = clean(block[4])
                if not text:
                    continue
                blocks.append({"ref": f"pdf:p{index}:b{block_number}", "text": text,
                               "bbox": [float(value) for value in block[:4]]})
            links = []
            for link_number, link in enumerate(page.get_links(), start=1):
                if link.get("uri"):
                    links.append({"ref": f"pdf:p{index}:link{link_number}",
                                  "uri": str(link["uri"]), "bbox": list(link["from"])})
            try:
                image_count = len(page.get_images(full=True))
            except Exception:
                image_count = None
            table_records = []
            table_detection_available = False
            try:
                finder = page.find_tables()
                table_detection_available = True
                for number, table in enumerate(finder.tables, start=1):
                    extracted = table.extract() or []
                    table_records.append({"ref": f"pdf:p{index}:table{number}",
                                          "rows": [[clean(cell) for cell in row] for row in extracted],
                                          "bbox": list(table.bbox) if table.bbox else None})
            except Exception:
                pass
            pages.append({"page_number": index, "width_pt": float(page.rect.width),
                          "height_pt": float(page.rect.height), "rotation": int(page.rotation),
                          "text": "\n".join(block["text"] for block in blocks), "blocks": blocks,
                          "links": links, "image_count": image_count, "tables": table_records,
                          "table_detection_available": table_detection_available,
                          "jpeg": _jpeg(page)})
        return {"pages": pages, "page_count": len(pages), "metadata": document.metadata or {}}
    finally:
        document.close()


def docx_evidence(docx_bytes: bytes | None, render: bool = True) -> dict[str, Any]:
    if not docx_bytes:
        return {"available": False, "units": [], "text": "", "table_rows": [],
                "image_count": 0, "rendered": False, "render_error": "DOCX not present"}
    try:
        from docx import Document
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = Document(io.BytesIO(docx_bytes))
        units = []
        table_rows = []
        for child in document.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, document)
                text = clean(paragraph.text)
                if text:
                    units.append({"ref": f"docx:u{len(units) + 1}", "kind": "paragraph",
                                  "style": paragraph.style.name if paragraph.style else "", "text": text})
            elif isinstance(child, CT_Tbl):
                table = Table(child, document)
                for row_index, row in enumerate(table.rows, start=1):
                    cells = [clean(cell.text) for cell in row.cells]
                    row_text = " | ".join(value for value in cells if value)
                    if row_text:
                        table_rows.append({"ref": f"docx:table{len(table_rows) + 1}:row{row_index}",
                                           "table_index": len(table_rows) + 1, "cells": cells, "text": row_text})
        images = sum(1 for rel in document.part.rels.values()
                     if rel.reltype.endswith("/image") and not rel.is_external)
        return _rendered_docx(docx_bytes, units, table_rows, images) if render else {
            "available": True, "units": units, "text": "\n".join(item["text"] for item in units),
            "table_rows": table_rows, "image_count": images, "rendered": False,
            "render_error": "DOCX rendering disabled"}
    except Exception as exc:
        return {"available": True, "units": [], "text": "", "table_rows": [], "image_count": 0,
                "rendered": False, "render_error": f"DOCX parse failed ({type(exc).__name__})"}


def _rendered_docx(docx_bytes: bytes, units: list[dict[str, Any]],
                   table_rows: list[dict[str, Any]], images: int) -> dict[str, Any]:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    base = {"available": True, "units": units, "text": "\n".join(item["text"] for item in units),
            "table_rows": table_rows, "image_count": images, "rendered": False, "render_error": None}
    if not executable:
        base["render_error"] = "LibreOffice/soffice is unavailable; rendered page alignment and visual checks are inconclusive."
        return base
    try:
        import pymupdf

        with tempfile.TemporaryDirectory(prefix="pdflens-docx-render-") as temp:
            root = Path(temp)
            source = root / "converted.docx"
            source.write_bytes(docx_bytes)
            result = subprocess.run([executable, "--headless", "--convert-to", "pdf",
                                     "--outdir", str(root), str(source)],
                                    capture_output=True, timeout=120, check=False)
            rendered_path = root / "converted.pdf"
            if result.returncode != 0 or not rendered_path.is_file():
                base["render_error"] = "LibreOffice could not render the DOCX."
                return base
            pdf = pymupdf.open(rendered_path)
            try:
                pages = []
                for index, page in enumerate(pdf, start=1):
                    pages.append({"page_number": index, "text": clean(page.get_text()),
                                  "width_pt": float(page.rect.width), "height_pt": float(page.rect.height),
                                  "jpeg": _jpeg(page)})
                base.update({"rendered": True, "render_error": None, "render_pages": pages,
                             "render_page_count": len(pages),
                             "renderer": Path(executable).name})
                return base
            finally:
                pdf.close()
    except Exception as exc:
        base["render_error"] = f"DOCX rendering failed ({type(exc).__name__})."
        return base


def data_url(raw: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
