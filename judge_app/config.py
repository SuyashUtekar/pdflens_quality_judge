from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Settings:
    max_package_bytes: int = _int_env("PDFLENS_JUDGE_MAX_PACKAGE_BYTES", 200 * 1024 * 1024)
    max_uncompressed_bytes: int = _int_env("PDFLENS_JUDGE_MAX_UNCOMPRESSED_BYTES", 800 * 1024 * 1024)
    max_zip_members: int = _int_env("PDFLENS_JUDGE_MAX_ZIP_MEMBERS", 4000)
    data_dir: Path = Path(os.getenv("PDFLENS_JUDGE_DATA_DIR", str(APP_ROOT / "data" / "evaluations")))
    schema_path: Path = Path(os.getenv("PDFLENS_METADATA_SCHEMA", str(APP_ROOT / "schemas" / "canonical-metadata-v1.schema.json")))
    llm_base_url: str | None = os.getenv("JUDGE_LLM_BASE_URL")
    llm_model: str | None = os.getenv("JUDGE_LLM_MODEL")
    llm_api_key: str | None = os.getenv("JUDGE_LLM_API_KEY")
    llm_timeout: int = _int_env("JUDGE_LLM_TIMEOUT_SECONDS", 120)
    llm_allow_remote: bool = os.getenv("JUDGE_LLM_ALLOW_REMOTE", "false").casefold() == "true"
    llm_json_mode: bool = os.getenv("JUDGE_LLM_JSON_MODE", "true").casefold() == "true"
    render_docx: bool = os.getenv("PDFLENS_JUDGE_RENDER_DOCX", "true").casefold() == "true"


settings = Settings()
