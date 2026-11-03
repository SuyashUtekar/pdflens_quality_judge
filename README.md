# PDFLens Quality Judge

A standalone MVP for evaluating PDFLens outputs against the original PDF. It has independent **Conversion** and **Metadata** modules, an Excel/JSON report, a FastAPI endpoint, a Streamlit upload interface, and an optional on-prem LiteLLM Proxy connection.

## Input

Upload the complete PDFLens ZIP. The ZIP must contain exactly one source PDF and may include `document.docx`, `metadata.json`, and image/table assets. If `manifest.json` exists, member SHA-256 hashes are checked. The metadata evaluator reads `processing.requested_fields`; if unavailable, field applicability is marked `INCONCLUSIVE` instead of assuming every field was requested. You can override it in the optional evaluation config.

The application validates the PDFLens canonical metadata v1 schema bundled at `schemas/canonical-metadata-v1.schema.json`. Set `PDFLENS_METADATA_SCHEMA` to a different approved schema file when evaluating another schema version.

## Start Streamlit

From this folder:

```sh
uv sync
make run-ui
```

Open `http://127.0.0.1:8502`, upload the PDFLens ZIP, select Conversion, Metadata, or both, then download the Excel or JSON report.

## Start the API

```sh
make run-api
```

Open `http://127.0.0.1:8020/docs`. Submit multipart fields `pdflens_package`, `modules` (`conversion`, `metadata`, or `conversion,metadata`), and optional `evaluation_config` JSON to `POST /api/v1/evaluations`.

## LiteLLM Proxy

The judge uses LiteLLM's OpenAI-compatible Chat Completions endpoint. Set these variables in the terminal before starting the UI or API:

```sh
export JUDGE_LLM_BASE_URL="http://<litellm-host>:4000/v1"
export JUDGE_LLM_MODEL="<model-alias-enabled-in-litellm>"
export JUDGE_LLM_API_KEY="<litellm-virtual-key>"
```

The base URL can be the proxy root (`http://<litellm-host>:4000`) or end in `/v1`. If the selected deployment does not support JSON response mode, set `JUDGE_LLM_JSON_MODE=false`. By default, remote/public endpoints are blocked; `JUDGE_LLM_ALLOW_REMOTE=true` should only be set for an approved endpoint. Prompt text and supplied page images/source evidence are sent to the configured gateway. Keep the API key out of source-controlled files.

## What this MVP checks

- **Shared ingestion:** ZIP integrity and size limits, manifest hashes, PDF signature, source/document/metadata member discovery, artifact references, and SHA-256 provenance.
- **Conversion:** PDF text blocks and page images; DOCX paragraphs/tables/images; optional DOCX-to-PDF rendering through locally installed LibreOffice; candidate page alignment using text similarity over contiguous rendered DOCX page spans; deterministic text coverage and numeric comparison; optional multimodal review of aligned PDF/DOCX rendered pages through LiteLLM.
- **Metadata:** canonical schema validation, requested-field applicability (`NOT_REQUESTED`), page coverage and numbering, heuristic body-text comparison, repeated header/footer candidates, PDF hyperlinks and metadata targets, table/image records and asset paths, duplicate element IDs, page bounds, and cross-field checks. Optional multimodal review is attempted for unresolved applicable items when configured.
- **Reports:** independent conversion and metadata scores/gates; `PASS`, `FAIL`, `N/A`, `NOT_REQUESTED`, and `INCONCLUSIVE`; criterion/page evidence references; Excel sheets for Summary, all Findings, module-specific findings, and Audit.

## Known MVP limits

- LibreOffice is optional. Without `soffice`/`libreoffice`, the application cannot render the DOCX, so page/region alignment and visual conversion criteria remain `INCONCLUSIVE`. It does not assume source PDF page N maps to DOCX page N.
- PDF table and image detectors have limitations. A raw image count does not establish which images are in scope; image/table findings that cannot be matched confidently remain inconclusive.
- Header/footer candidate detection uses repeated text in top/bottom page bands. Single-page or non-repeating furniture may require multimodal/human review.
- Some criteria remain explicitly inconclusive until reliable evidence, rendering, or human-reviewed calibration data is available. Thresholds and matching cutoffs are provisional, not release standards.
- Evaluations run synchronously and reports are stored locally under `data/evaluations/` for the API. Streamlit keeps report artifacts in the active browser session for download. Authentication, database persistence, queueing, and production deployment hardening are not implemented.
- The app does not modify PDFLens output artifacts.

## Optional settings

| Variable | Purpose | Default |
|---|---|---|
| `PDFLENS_JUDGE_MAX_PACKAGE_BYTES` | Maximum uploaded ZIP size | 200 MiB |
| `PDFLENS_JUDGE_MAX_UNCOMPRESSED_BYTES` | Maximum expanded ZIP size | 800 MiB |
| `PDFLENS_JUDGE_MAX_ZIP_MEMBERS` | Maximum ZIP entries | 4000 |
| `PDFLENS_JUDGE_DATA_DIR` | API report storage directory | `data/evaluations` |
| `PDFLENS_METADATA_SCHEMA` | Metadata JSON Schema path | bundled canonical v1 schema |
| `PDFLENS_JUDGE_RENDER_DOCX` | Attempt local LibreOffice DOCX rendering | `true` |

