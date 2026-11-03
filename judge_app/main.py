from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import settings
from .evaluation import evaluate
from .ingestion import InputError, parse_config, read_package
from .reporting import build_xlsx, report_json_bytes

logger = logging.getLogger("pdflens_quality_judge")
app = FastAPI(title="PDFLens Quality Judge", version="0.1.0",
              description="Independent conversion and metadata quality evaluation for PDFLens output.")


def _directory(evaluation_id: str) -> Path:
    if not re.fullmatch(r"pdflens-eval-[0-9a-f-]{36}", evaluation_id):
        raise HTTPException(404, detail={"code": "EVALUATION_NOT_FOUND", "message": "Evaluation not found."})
    path = settings.data_dir / evaluation_id
    if not path.is_dir():
        raise HTTPException(404, detail={"code": "EVALUATION_NOT_FOUND", "message": "Evaluation not found."})
    return path


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pdflens-quality-judge"}


@app.post("/api/v1/evaluations", status_code=201)
async def create_evaluation(
    pdflens_package: UploadFile = File(...),
    modules: str = Form("conversion,metadata"),
    evaluation_config: str = Form("{}"),
) -> JSONResponse:
    raw = await pdflens_package.read(settings.max_package_bytes + 1)
    if len(raw) > settings.max_package_bytes:
        raise HTTPException(413, detail={"code": "PACKAGE_TOO_LARGE", "message": "ZIP exceeds the configured size."})
    selected = [item.strip().casefold() for item in modules.split(",") if item.strip()]
    if not selected or set(selected) - {"conversion", "metadata"}:
        raise HTTPException(422, detail={"code": "INVALID_MODULES", "message": "Choose conversion, metadata, or both."})
    try:
        run_input = read_package(raw)
        config = parse_config(evaluation_config)
    except InputError as exc:
        raise HTTPException(422, detail={"code": "INVALID_INPUT", "message": str(exc)}) from exc
    try:
        report = await run_in_threadpool(evaluate, run_input, selected, config)
        report["pdflens_package"]["uploaded_filename"] = Path(pdflens_package.filename or "pdflens-output.zip").name
        report["configuration"] = {key: value for key, value in config.items() if key != "entity_map"}
        report_json = report_json_bytes(report)
        workbook = await run_in_threadpool(build_xlsx, report)
        target = settings.data_dir / report["evaluation_id"]
        target.mkdir(parents=True, exist_ok=False)
        (target / "report.json").write_bytes(report_json)
        (target / "report.xlsx").write_bytes(workbook)
    except Exception as exc:
        request_id = str(uuid.uuid4())
        logger.exception("quality evaluation failed request_id=%s type=%s", request_id, type(exc).__name__)
        raise HTTPException(500, detail={"code": "EVALUATION_FAILED", "request_id": request_id,
                                         "message": f"Evaluation failed ({type(exc).__name__})."}) from exc
    return JSONResponse(status_code=201, content={
        "evaluation_id": report["evaluation_id"], "status": report["status"],
        "overall_quality_gate": report["overall_quality_gate"], "module_reports": report["module_reports"],
        "report_url": f"/api/v1/evaluations/{report['evaluation_id']}/report",
        "workbook_url": f"/api/v1/evaluations/{report['evaluation_id']}/report.xlsx",
    })


@app.get("/api/v1/evaluations/{evaluation_id}/report")
def get_report(evaluation_id: str) -> dict[str, Any]:
    try:
        return json.loads((_directory(evaluation_id) / "report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, detail={"code": "REPORT_UNAVAILABLE", "message": "Report unavailable."}) from exc


@app.get("/api/v1/evaluations/{evaluation_id}/report.xlsx")
def get_workbook(evaluation_id: str) -> FileResponse:
    path = _directory(evaluation_id) / "report.xlsx"
    if not path.is_file():
        raise HTTPException(404, detail={"code": "REPORT_NOT_FOUND", "message": "Workbook unavailable."})
    return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename=f"{evaluation_id}_pdflens_quality_report.xlsx")


@app.on_event("startup")
def initialize_storage() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
