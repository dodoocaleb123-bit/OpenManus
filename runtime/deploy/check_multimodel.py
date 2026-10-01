"""Read-only Ollama probe: python -m deploy.check_multimodel inside the container.

No project writes, GitHub calls, paid APIs, or printed prompts/responses/secrets.
Models are checked and invoked one at a time, then asked to unload from Ollama.
"""
from __future__ import annotations

import asyncio
import base64
import ipaddress
import os
import time
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit

import httpx
from PIL import Image

from app.config import config
from app.llm import LLM, strip_think_tags
from app.platform.model_health import check_model_role
from app.platform.reasoning import make_authoritative_plan

ROLES = (
    ("reasoning", "DeepSeek"),
    ("heavy_coding", "Qwen Coder"),
    ("vision", "Gemma 3"),
    ("research", "Qwen2.5 3B"),
    ("creativity", "Llama 3.2 3B"),
)


def _local_host(base_url: str) -> bool:
    host = (urlsplit(base_url).hostname or "").lower()
    if host in {"host.docker.internal", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _ollama_generate_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    prefix = parts.path.rstrip("/")
    if prefix.endswith("/v1"):
        prefix = prefix[:-3]
    return urlunsplit((parts.scheme, parts.netloc, prefix + "/api/generate", "", ""))


async def _unload(settings) -> None:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                _ollama_generate_url(settings.base_url),
                json={"model": settings.model, "prompt": "", "keep_alive": 0, "stream": False},
            )
    except Exception:
        print("  Note: Ollama did not confirm model unload; allow memory to recover before the next run.", flush=True)


def _red_image() -> str:
    picture = Image.new("RGB", (32, 32), (255, 0, 0))
    buffer = BytesIO()
    picture.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


async def _probe(role: str) -> None:
    model = LLM(config_name=role)
    if role == "reasoning":
        plan = await make_authoritative_plan(
            model, prompt="Hellooo", context={"available_model_roles": list(config.llm)}
        )
        if plan["route"] != "respond":
            raise ValueError("DeepSeek did not choose a direct response for the greeting")
        answer = await model.ask(
            [{"role": "user", "content": "Hellooo"}],
            system_msgs=[{"role": "system", "content": "Respond naturally to this user greeting."}],
            stream=False, max_tokens=900,
        )
    elif role == "vision":
        answer = await model.ask_with_images(
            [{"role": "user", "content": "What color is the attached solid-color square?"}],
            [_red_image()], stream=False, max_tokens=140,
        )
        if "red" not in strip_think_tags(answer).casefold():
            raise ValueError("Gemma returned no image-grounded color answer")
    else:
        answer = await model.ask(
            [{"role": "user", "content": "Respond with one short sentence confirming you are available."}],
            stream=False, max_tokens=240,
        )
    if not strip_think_tags(answer).strip():
        raise ValueError("Model returned no visible answer")


async def main() -> int:
    print("OpenManus local Ollama five-model smoke test (no credentials or answers are printed)", flush=True)
    timeout = min(900, max(60, int(os.environ.get("OPENMANUS_SMOKE_TIMEOUT_SECONDS", "420"))))
    settings_by_role = {}
    missing = False
    for role, label in ROLES:
        settings = config.llm.get(role) or (config.llm.get("default") if role == "heavy_coding" else None)
        if settings is None or not _local_host(settings.base_url):
            print(f"{label}: NOT CONFIGURED FOR LOCAL OLLAMA", flush=True)
            missing = True
            continue
        settings_by_role[role] = settings
        health = await check_model_role(role)
        if not health.get("reachable") or not health.get("model_present"):
            print(f"{label}: MISSING OR UNREACHABLE ({health.get('error') or 'not installed'})", flush=True)
            missing = True
        else:
            print(f"{label}: available ({settings.model})", flush=True)
    if missing:
        print("Result: FAIL; check the .env model-role tags, `ollama list`, and host.docker.internal connectivity.", flush=True)
        return 1

    failed = False
    for role, label in ROLES:
        started = time.monotonic()
        print(f"{label}: probing sequentially...", flush=True)
        try:
            await asyncio.wait_for(_probe(role), timeout=timeout)
            print(f"{label}: PASS ({time.monotonic() - started:.1f}s)", flush=True)
        except Exception as exc:
            print(f"{label}: FAIL ({type(exc).__name__}; {time.monotonic() - started:.1f}s)", flush=True)
            failed = True
        finally:
            await _unload(settings_by_role[role])
    print("Result: " + ("FAIL" if failed else "PASS (five model calls + DeepSeek greeting; not the full build/preview scenario)"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
