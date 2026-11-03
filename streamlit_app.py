from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from judge_app.evaluation import evaluate
from judge_app.ingestion import InputError, parse_config, read_package
from judge_app.reporting import build_xlsx, report_json_bytes

logger = logging.getLogger("pdflens_quality_judge.streamlit")
st.set_page_config(page_title="PDFLens Quality Judge", page_icon="📄", layout="wide")
st.title("PDFLens Quality Judge")
st.write("Evaluate PDFLens conversion and metadata outputs against the original PDF in its ZIP package.")

with st.sidebar:
    st.subheader("Evaluation setup")
    modules = st.multiselect("Evaluation modules", ["conversion", "metadata"],
                             default=["conversion", "metadata"])
    config_text = st.text_area(
        "Evaluation configuration (optional JSON)", value="{}", height=180,
        help=("Use requested_fields to override metadata.processing.requested_fields, "
              "and set provisional module thresholds if needed."),
    )
    st.caption("LiteLLM settings are supplied through environment variables before starting Streamlit.")

uploaded = st.file_uploader("Complete PDFLens output ZIP", type=["zip"])
if uploaded:
    st.caption(f"{uploaded.name} · {uploaded.size / (1024 * 1024):.1f} MiB")
run_button = st.button("Run quality evaluation", type="primary", disabled=not uploaded or not modules,
                       use_container_width=True)

if run_button:
    for key in ("quality_report", "quality_xlsx", "quality_json"):
        st.session_state.pop(key, None)
    try:
        run_input = read_package(uploaded.getvalue())
        config = parse_config(config_text)
        with st.spinner("Building PDF evidence and evaluating selected modules…"):
            report = evaluate(run_input, modules, config)
            report["pdflens_package"]["uploaded_filename"] = Path(uploaded.name).name
            report["configuration"] = {key: value for key, value in config.items() if key != "entity_map"}
            st.session_state["quality_report"] = report
            st.session_state["quality_xlsx"] = build_xlsx(report)
            st.session_state["quality_json"] = report_json_bytes(report)
    except InputError as exc:
        st.error(str(exc))
    except Exception as exc:
        logger.exception("evaluation failed error_type=%s", type(exc).__name__)
        st.error(f"Evaluation failed ({type(exc).__name__}). See the terminal for the diagnostic log.")

report: dict[str, Any] | None = st.session_state.get("quality_report")
if report:
    st.divider()
    st.subheader("Quality gates")
    gate = report["overall_quality_gate"]
    if gate == "PASS":
        st.success(f"Overall: {gate}")
    elif gate == "FAIL":
        st.error(f"Overall: {gate}")
    else:
        st.warning(f"Overall: {gate}")
    module_columns = st.columns(max(1, len(report["module_reports"])))
    for column, (module, result) in zip(module_columns, report["module_reports"].items()):
        column.metric(f"{module.title()} gate", result["quality_gate"])
        column.metric("Pass percentage", f"{result['pass_percentage']:.2f}%" if result["pass_percentage"] is not None else "—")
        column.caption(f"Scorable {result['scorable_count']} · FAIL {result['fail_count']} · INCONCLUSIVE {result['inconclusive_count']}")
    st.caption(f"Evaluation `{report['evaluation_id']}` · LLM configured: {'yes' if report['llm']['configured'] else 'no'}")
    if "conversion" in report["selected_modules"] and report["conversion"].get("docx_render_error"):
        st.info(report["conversion"]["docx_render_error"])

    stem = Path(report.get("document_id") or Path(report["source_pdf"]["file_name"]).stem).stem
    left, right = st.columns(2)
    left.download_button("Download Excel report", st.session_state["quality_xlsx"],
                         file_name=f"{stem}_pdflens_quality_report.xlsx",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                         type="primary", use_container_width=True)
    right.download_button("Download JSON report", st.session_state["quality_json"],
                          file_name=f"{stem}_pdflens_quality_report.json", mime="application/json",
                          use_container_width=True)

    findings = report["findings"]
    tabs = st.tabs(["Findings", "Conversion alignment", "Metadata audit", "Run details"])
    with tabs[0]:
        selected_module = st.selectbox("Module", report["selected_modules"])
        outcomes = st.multiselect("Outcomes", ["FAIL", "INCONCLUSIVE", "PASS", "N/A", "NOT_REQUESTED"],
                                 default=["FAIL", "INCONCLUSIVE"])
        visible = [item for item in findings if item["module"] == selected_module and item["outcome"] in outcomes]
        st.dataframe(visible, use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(report.get("conversion", {}).get("page_alignment", []), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.json(report.get("metadata", {}))
    with tabs[3]:
        st.json({key: value for key, value in report.items() if key != "findings"})
    st.info("Criteria, heuristics, and quality-gate thresholds are provisional. Review INCONCLUSIVE results and evidence before acceptance.")
