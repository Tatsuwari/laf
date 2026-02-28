# laf/sandbox/runner.py
from __future__ import annotations

import importlib
import json
import sys
import traceback
from typing import Any, Dict


def _block_network() -> None:
    """
    Best-effort network disable: monkeypatch socket creation.
    This blocks most network libs (requests, urllib3, etc.).
    """
    import socket

    class _NoNetSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Network is disabled by sandbox policy.")

    socket.socket = _NoNetSocket  # type: ignore


def _import_by_entrypoint(ep: str):
    # "package.module:function_name"
    if ":" not in ep:
        raise ValueError(f"Invalid entrypoint '{ep}'. Expected 'module:function'.")
    mod_name, fn_name = ep.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise AttributeError(f"Function '{fn_name}' not found in '{mod_name}'.")
    return fn


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"ok": False, "error": "empty_payload"}))
        return 2

    try:
        payload: Dict[str, Any] = json.loads(raw)
        entrypoint = payload["entrypoint"]
        args = payload.get("args", {}) or {}
        allow_network = bool(payload.get("allow_network", False))

        if not allow_network:
            _block_network()

        fn = _import_by_entrypoint(entrypoint)

        if not isinstance(args, dict):
            raise ValueError("args must be a JSON object")

        result = fn(**args)

        print(json.dumps({"ok": True, "result": result}))
        return 0

    except Exception as e:
        tb = traceback.format_exc(limit=10)
        print(json.dumps({"ok": False, "error": str(e), "traceback": tb}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())