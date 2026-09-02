#!/usr/bin/env python3
"""Run one release gate and retain bounded, path-sanitized stdout/stderr."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


MAX_STREAM_BYTES = 900_000


def _sanitize(text: str, root: Path) -> str:
    text = text.replace(str(root), ".")
    text = re.sub(r"/(?:Users|home)/[^\s\"']+", "<LOCAL_PATH>", text)
    text = re.sub(r"/private/(?:tmp|var/folders)/[^\s\"']+", "<TEMP_PATH>", text)
    return text


def _bounded(text: str) -> str:
    data = text.encode("utf-8")
    if len(data) > MAX_STREAM_BYTES:
        raise RuntimeError(f"release command output exceeds {MAX_STREAM_BYTES} bytes")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--parser", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    root = Path(__file__).resolve().parents[2]
    environment: dict[str, str] = {}
    child_env = os.environ.copy()
    for item in args.env:
        if "=" not in item:
            parser.error("--env values must be NAME=value")
        name, value = item.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            parser.error("--env names must be upper-case identifiers")
        environment[name] = value
        child_env[name] = value
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=root,
        env=child_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    duration = round(time.monotonic() - started, 3)
    payload = {
        "schema": "worldzero-release-command-log-v1",
        "id": args.id,
        "argv": command,
        "cwd": ".",
        "environment": dict(sorted(environment.items())),
        "parser": args.parser,
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "stdout": _bounded(_sanitize(completed.stdout, root)),
        "stderr": _bounded(_sanitize(completed.stderr, root)),
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.log.with_name(f".{args.log.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.log)
    print(json.dumps({
        "id": args.id,
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "log": args.log.as_posix(),
    }, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
