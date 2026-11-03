.PHONY: install run-api run-ui

install:
	uv sync

run-api:
	uv run uvicorn judge_app.main:app --reload --host 127.0.0.1 --port 8020

run-ui:
	uv run streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8502
