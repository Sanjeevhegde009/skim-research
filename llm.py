"""
LLM plumbing — provider-agnostic call layer for the compiler and query models.

Extracted from the original hlma.py so the PageIndex + RAG + gate stack has no
HLMA dependency. Two roles, driven by config.py:
  - compiler_call(...) — the frontier "compiler"/judge model (anthropic / openai / openai_compatible)
  - query_call(...)    — the reader model (ollama / anthropic / openai / openai_compatible)
Keys are read from environment variables only (never stored in files).
"""

import json
import os
import time

import requests

import config


def _get_api_key(env_var):
    key = os.environ.get(env_var, "")
    if not key:
        print(f"  [ERROR] Set {env_var} environment variable")
    return key


def _api_call_with_retry(fn, max_retries=5):
    """Retry an API call with exponential backoff on 429 / transient errors."""
    delay = 10
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            is_431        = "431" in msg or "Request Header Fields Too Large" in msg
            is_rate_limit = "429" in msg or "Too Many Requests" in msg
            is_transient  = "500" in msg or "502" in msg or "503" in msg or "timeout" in msg.lower()
            # Connection-level failures (network blip, DNS hiccup, server hangup)
            # are as transient as a 500 — a dropped call here silently loses facts.
            is_network    = isinstance(e, (ConnectionError, OSError)) or any(
                s in msg for s in ("Connection aborted", "RemoteDisconnected",
                                   "NameResolutionError", "getaddrinfo",
                                   "Connection refused", "Connection reset",
                                   "ConnectionError", "Max retries exceeded"))
            if is_431 or is_rate_limit or is_transient or is_network:
                body = ""
                if hasattr(e, "response") and e.response is not None:
                    try:
                        body = e.response.json().get("error", {}).get("message", "")
                    except Exception:
                        body = e.response.text[:400]
                print(f"  [RATE LIMIT] {body or msg}")
                if attempt < max_retries - 1:
                    # 431 is edge flakiness, not rate limiting — a short fixed wait
                    # suffices; exponential backoff just stalls compilation.
                    wait = 2 if is_431 else delay * (2 ** attempt)
                    print(f"  [RETRY {attempt+1}/{max_retries-1}] waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise


def _call_anthropic(prompt, system, temperature, model, api_key_env):
    key = _get_api_key(api_key_env)
    if not key: return ""
    messages = [{"role": "user", "content": prompt}]
    body = {"model": model, "max_tokens": 4096, "temperature": temperature, "messages": messages}
    if system: body["system"] = system
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


def _call_openai_compat(prompt, system, temperature, model, api_key_env, provider, base_url=""):
    key = _get_api_key(api_key_env)
    if not key: return ""
    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
    else:
        url = base_url
    if not url:
        print("  [ERROR] Set base URL"); return ""
    messages = []
    if system: messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "temperature": temperature, "max_tokens": 4096, "messages": messages}
    resp = requests.post(url, headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"}, json=body, timeout=300)
    if resp.status_code == 431:
        # OpenAI's edge sporadically mislabels requests as "headers too large".
        # Log what was actually sent so the real trigger is identifiable.
        print(f"  [431 DIAG] body_bytes={len(json.dumps(body))} "
              f"key_len={len(key)} model={model} "
              f"prompt_chars={sum(len(m['content']) for m in messages)}")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# Count of compiler calls that hard-failed (exhausted retries). Callers can check
# this around compilation: any failure means results have holes and must NOT be
# cached as complete.
COMPILER_FAILURES = 0


def compiler_call(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    """Call compiler model. Temperature defaults to 0.0 for determinism."""
    global COMPILER_FAILURES
    try:
        if config.COMPILER_PROVIDER == "anthropic":
            return _api_call_with_retry(lambda: _call_anthropic(
                prompt, system, temperature,
                config.COMPILER_MODEL, config.COMPILER_API_KEY_ENV))
        else:
            return _api_call_with_retry(lambda: _call_openai_compat(
                prompt, system, temperature,
                config.COMPILER_MODEL, config.COMPILER_API_KEY_ENV,
                config.COMPILER_PROVIDER, config.COMPILER_BASE_URL))
    except Exception as e:
        COMPILER_FAILURES += 1
        print(f"  [COMPILER ERROR] {e}")
        return ""


def query_call(messages: list, temperature: float = 0.1) -> str:
    """Call query model — Ollama or API. Retries on 429 / transient errors."""
    def _do_call():
        if config.QUERY_PROVIDER == "ollama":
            resp = requests.post(
                config.OLLAMA_URL,
                json={"model": config.QUERY_MODEL, "messages": messages,
                      "stream": False,
                      "think": False,
                      "options": {"temperature": temperature}},
                timeout=600)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        elif config.QUERY_PROVIDER == "anthropic":
            key = _get_api_key(config.QUERY_API_KEY_ENV)
            if not key: return ""
            system = ""
            user_content = ""
            for m in messages:
                if m["role"] == "system": system = m["content"]
                else: user_content += m["content"] + "\n"
            body = {"model": config.QUERY_MODEL, "max_tokens": 1024,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": user_content.strip()}]}
            if system: body["system"] = system
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json=body, timeout=120)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"].strip()
        else:
            key = _get_api_key(config.QUERY_API_KEY_ENV)
            if not key: return ""
            url = "https://api.openai.com/v1/chat/completions" if config.QUERY_PROVIDER == "openai" else config.QUERY_BASE_URL
            body = {"model": config.QUERY_MODEL, "temperature": temperature,
                    "max_tokens": 1024, "messages": messages}
            resp = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                 "Content-Type": "application/json"}, json=body, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    try:
        return _api_call_with_retry(_do_call)
    except Exception as e:
        print(f"  [QUERY ERROR] {e}")
        return ""


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)
