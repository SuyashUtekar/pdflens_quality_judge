from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .config import settings


class InputError(ValueError):
    """Uploaded PDFLens artifacts could not be safely normalized."""


@dataclass
class RunInput:
    pdf_bytes: bytes
    docx_bytes: bytes | None
    metadata_raw: bytes | None
    metadata: dict[str, Any] | None
    metadata_parse_error: str | None
    assets: dict[str, bytes]
    members: list[str]
    pdf_name: str
    docx_name: str | None
    package_sha256: str


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts and "\\" not in name
            and "\x00" not in name)


def _check_manifest(archive: zipfile.ZipFile, files: dict[str, zipfile.ZipInfo]) -> None:
    manifests = [name for name in files if PurePosixPath(name).name.casefold() == "manifest.json"]
    if len(manifests) > 1:
        raise InputError("The ZIP contains multiple manifest.json files.")
    if not manifests:
        return
    try:
        manifest = json.loads(archive.read(manifests[0]))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError("manifest.json is invalid JSON.") from exc
    entries = manifest.get("members", []) if isinstance(manifest, dict) else []
    if not isinstance(entries, list):
        raise InputError("Manifest members must be an array.")
    for entry in entries:
        if not isinstance(entry, dict):
            raise InputError("Manifest contains an invalid member record.")
        name = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(name, str) or name not in files or not _safe_name(name):
            raise InputError("Manifest references an unsafe or missing artifact.")
        if expected and sha256(archive.read(name)).casefold() != str(expected).casefold():
            raise InputError(f"Manifest checksum mismatch for {name}.")


def read_package(raw: bytes) -> RunInput:
    if len(raw) > settings.max_package_bytes:
        raise InputError("PDFLens ZIP exceeds the configured upload limit.")
    stream = io.BytesIO(raw)
    if not zipfile.is_zipfile(stream):
        raise InputError("Upload a valid PDFLens ZIP package.")
    try:
        with zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            if len(infos) > settings.max_zip_members:
                raise InputError("PDFLens ZIP contains too many entries.")
            if sum(info.file_size for info in infos) > settings.max_uncompressed_bytes:
                raise InputError("PDFLens ZIP expands beyond the configured size limit.")
            names = [info.filename for info in infos]
            if any(not _safe_name(name) for name in names):
                raise InputError("PDFLens ZIP contains an unsafe member path.")
            if len({name.casefold() for name in names}) != len(names):
                raise InputError("PDFLens ZIP contains duplicate member names.")
            if archive.testzip() is not None:
                raise InputError("PDFLens ZIP contains a corrupt member.")
            files = {info.filename: info for info in infos if not info.is_dir()}
            pdfs = [name for name in files if name.casefold().endswith(".pdf")]
            docxs = [name for name in files if name.casefold().endswith(".docx")]
            metadatas = [name for name in files if PurePosixPath(name).name.casefold() == "metadata.json"]
            if len(pdfs) != 1:
                raise InputError("PDFLens ZIP must contain exactly one source PDF.")
            if len(docxs) > 1 or len(metadatas) > 1:
                raise InputError("PDFLens ZIP must contain at most one DOCX and one metadata.json.")
            _check_manifest(archive, files)
            pdf_name = pdfs[0]
            pdf_bytes = archive.read(pdf_name)
            if not pdf_bytes.startswith(b"%PDF-"):
                raise InputError("Source PDF signature is invalid.")
            docx_name = docxs[0] if docxs else None
            docx_bytes = archive.read(docx_name) if docx_name else None
            metadata_raw = archive.read(metadatas[0]) if metadatas else None
            metadata = None
            parse_error = None
            if metadata_raw is not None:
                try:
                    parsed = json.loads(metadata_raw)
                    if isinstance(parsed, dict):
                        metadata = parsed
                    else:
                        parse_error = "metadata.json root must be a JSON object."
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    parse_error = f"metadata.json parsing failed: {type(exc).__name__}."
            excluded = {pdf_name, docx_name, *metadatas}
            assets = {name: archive.read(name) for name in files
                      if name not in excluded and not name.casefold().endswith(".json")}
            return RunInput(pdf_bytes, docx_bytes, metadata_raw, metadata, parse_error,
                            assets, names, pdf_name, docx_name, sha256(raw))
    except zipfile.BadZipFile as exc:
        raise InputError("PDFLens ZIP is corrupt or unreadable.") from exc


def parse_config(raw: str | None) -> dict[str, Any]:
    if raw and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InputError("Evaluation configuration must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise InputError("Evaluation configuration must be a JSON object.")
        for field in ("conversion_pass_candidate_at", "conversion_review_required_below",
                      "metadata_pass_candidate_at", "metadata_review_required_below"):
            if field in payload:
                value = payload[field]
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 100:
                    raise InputError(f"{field} must be a number between 0 and 100.")
        for module in ("conversion", "metadata"):
            passing = payload.get(f"{module}_pass_candidate_at", 95)
            review = payload.get(f"{module}_review_required_below", 85)
            if review >= passing:
                raise InputError(f"{module} review threshold must be lower than its pass threshold.")
        for field in ("requested_fields", "extraction_fields"):
            value = payload.get(field)
            if value is not None and (not isinstance(value, list)
                                      or not all(isinstance(item, str) for item in value)):
                raise InputError(f"{field} must be an array of strings.")
        return payload
    return {}
