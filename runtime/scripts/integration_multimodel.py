#!/usr/bin/env python3
"""Smoke-test a running OpenManus container and its five Ollama roles.

Usage: python scripts/integration_multimodel.py [base_url]
The script never sends credentials or model prompts to logs.
"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
EXPECTED = {"qwen_coder", "gemma3", "deepseek", "qwen2.5_3b", "llama3.2_3b"}


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=15) as response:
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}")
        return json.load(response)


health = get("/api/health")
if health.get("status") != "ok":
    raise SystemExit(f"health failed: {health}")
registry = get("/api/capabilities/registry")
if registry.get("count") != 400:
    raise SystemExit(f"registry count failed: {registry.get('count')}")
if registry.get("schema_version") != "1.2.0" or not all("input_schema" in item and "output_schema" in item for item in registry.get("capabilities", [])):
    raise SystemExit("registry schema contract failed")
roles = get("/api/capabilities/models/status").get("models", [])
seen = {item.get("role") for item in roles}
if seen != EXPECTED:
    raise SystemExit(f"role map failed: {sorted(seen)}")
deepseek = next(item for item in roles if item.get("role") == "deepseek")
if not deepseek.get("enabled") or not deepseek.get("planning"):
    raise SystemExit("DeepSeek planner role is not enabled")
health_roles = get("/api/capabilities/models/health").get("models", [])
failed = [item for item in health_roles if not item.get("reachable") or not item.get("model_present")]
if failed:
    raise SystemExit("model health failed: " + ", ".join(f"{item.get('role')}:{item.get('error')}" for item in failed))
print(json.dumps({"health": health, "registry_count": registry["count"], "roles": sorted(seen), "health_checks": len(health_roles)}, indent=2))
print("OpenManus multi-model integration contract: PASS")
