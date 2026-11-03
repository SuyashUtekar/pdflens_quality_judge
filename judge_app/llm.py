from __future__ import annotations

import json
import re
import socket
from ipaddress import ip_address, ip_network
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

import httpx

from .config import settings


class LLMError(RuntimeError):
    """The approved on-prem judge endpoint is unavailable or returned invalid output."""


def _private_host(host: str) -> bool:
    host = host.casefold().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal", ".lan")):
        return True
    try:
        addresses = {ip_address(host)}
    except ValueError:
        try:
            addresses = {ip_address(item[4][0]) for item in socket.getaddrinfo(host, None)}
        except OSError:
            return False
    networks = (ip_network("10.0.0.0/8"), ip_network("172.16.0.0/12"),
                ip_network("192.168.0.0/16"), ip_network("127.0.0.0/8"),
                ip_network("fc00::/7"), ip_network("::1/128"))
    return bool(addresses) and all(any(address in network for network in networks)
                                   for address in addresses)


class LiteLLMJudge:
    def __init__(self) -> None:
        if not settings.llm_base_url or not settings.llm_model:
            raise LLMError("JUDGE_LLM_BASE_URL and JUDGE_LLM_MODEL are not configured.")
        parsed = urlparse(settings.llm_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise LLMError("LiteLLM base URL must use HTTP or HTTPS.")
        if not settings.llm_allow_remote and not _private_host(parsed.hostname):
            raise LLMError("LiteLLM host is not recognized as local/private; remote hosts are disabled.")
        endpoint = settings.llm_base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions" if parsed.path.rstrip("/").endswith("/v1") else "/v1/chat/completions"
        self.endpoint = endpoint
        self.model = settings.llm_model

    def judge(self, prompt_file: str, payload: dict[str, Any], images: list[bytes]) -> dict[str, Any]:
        prompt = (Path(__file__).resolve().parents[1] / "prompts" / prompt_file).read_text(encoding="utf-8")
        content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        for image in images:
            import base64
            encoded = base64.b64encode(image).decode("ascii")
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}})
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        body: dict[str, Any] = {
            "model": self.model, "temperature": 0,
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": content}],
        }
        if settings.llm_json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            with httpx.Client(timeout=settings.llm_timeout, follow_redirects=False) as client:
                response = client.post(self.endpoint, json=body, headers=headers)
                response.raise_for_status()
            message = response.json()["choices"][0]["message"]["content"]
            if not isinstance(message, str):
                raise ValueError("Expected text JSON response")
            message = re.sub(r"^```(?:json)?\s*|\s*```$", "", message.strip(), flags=re.I)
            result = json.loads(message)
            if not isinstance(result, dict) or not isinstance(result.get("criteria"), list):
                raise ValueError("Response must contain a criteria array")
            return result
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError(f"LiteLLM call or response validation failed ({type(exc).__name__}).") from exc
