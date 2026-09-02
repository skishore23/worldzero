#!/usr/bin/env python3
"""Execute the non-paid README quick start from a clean public source copy."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable


FAMILY_IDS = (
    "worldzero:catalysis",
    "worldzero:inhibition",
    "worldzero:delayed-transformation",
    "worldzero:null",
)
WARNING = (
    "WARNING: experimental law-family plugins are trusted in-process Python and may "
    "execute arbitrary code. Loading selected exact entry point: example_org:preserver"
)
DETACHED_FILES = {"release-verification.json"}
DETACHED_PREFIXES = ("evidence/release/commands/",)


def _copy_public(root: Path, destination: Path) -> None:
    manifest = json.loads((root / "docs/public-files.json").read_text(encoding="utf-8"))
    copied: list[str] = []
    for name in manifest["files"]:
        if name in DETACHED_FILES or any(name.startswith(prefix) for prefix in DETACHED_PREFIXES):
            continue
        source = root / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(name)
    (destination / "docs/public-files.json").write_text(
        json.dumps({"schema": "worldzero-public-files-v1", "files": copied}, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalized_readme(text: str) -> str:
    joined = re.sub(r"\\\s*\n\s*", " ", text)
    return re.sub(r"[ \t]+", " ", joined)


def _required_readme_commands() -> tuple[str, ...]:
    commands = [
        "python -m venv .venv",
        "python -m pip install --no-build-isolation -e '.[test]'",
        "python -m worldzero laws list",
        *(f"python -m worldzero laws inspect {family_id}" for family_id in FAMILY_IDS),
        *(f"python -m worldzero laws validate {family_id} --seeds 1" for family_id in FAMILY_IDS),
        "python -m worldzero demo --seeds 1 --output worldzero-demo",
        "python -m worldzero replay worldzero-demo/traces/pressure-experimenter/1452232541.json.gz",
        "python -m worldzero benchmark create-manifest --output benchmark.json --dev-count 1 --test-count 1",
        "python -m worldzero benchmark run --manifest benchmark.json --agent examples.custom_agent:create_agent --agent-version 0.1.0 --split dev --no-baselines --output runs/custom-agent",
        "python -m pip wheel --no-deps --no-build-isolation --wheel-dir example-dist ./examples/community_law_plugin",
        "python -m pip install --no-index --no-deps example-dist/worldzero_example_law-0.1.0-py3-none-any.whl",
        "python -m worldzero laws inspect example_org:preserver --experimental-family",
        "python -m worldzero laws validate example_org:preserver --experimental-family --seeds 1",
        "python -m worldzero laws inspect example_org:preserver",
        "python -m worldzero demo --seeds 1 --law-family example_org:preserver --experimental-family --output worldzero-example-demo",
        "python -m worldzero replay worldzero-example-demo/traces/pressure-experimenter/1452232541.json.gz",
    ]
    return tuple(commands)


def _json(stdout: str) -> dict[str, object]:
    value = json.loads(stdout)
    if type(value) is not dict:
        raise RuntimeError("command output is not a JSON object")
    return value


def _run_check(
    checks: list[dict[str, object]],
    *,
    name: str,
    argv: list[str],
    cwd: Path,
    python: Path,
    env: dict[str, str],
    expected_exit_code: int = 0,
    validate: Callable[[subprocess.CompletedProcess[str]], bool] | None = None,
) -> subprocess.CompletedProcess[str]:
    actual = [str(python), *argv[1:]] if argv[0] == "python" else argv
    started = time.monotonic()
    result = subprocess.run(
        actual,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    duration = round(time.monotonic() - started, 3)
    passed = result.returncode == expected_exit_code and (validate(result) if validate else True)
    checks.append({
        "name": name,
        "argv": argv,
        "exit_code": result.returncode,
        "expected_exit_code": expected_exit_code,
        "duration_seconds": duration,
        "passed": passed,
    })
    if not passed:
        raise RuntimeError(
            f"README check failed: {name}; exit={result.returncode}; "
            f"stdout={result.stdout[-1000:]!r}; stderr={result.stderr[-1000:]!r}"
        )
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    checks: list[dict[str, object]] = []
    readme = _normalized_readme((root / "README.md").read_text(encoding="utf-8"))
    missing = [command for command in _required_readme_commands() if command not in readme]
    checks.append({
        "name": "readme_command_manifest",
        "argv": ["README.md"],
        "exit_code": 0 if not missing else 1,
        "expected_exit_code": 0,
        "duration_seconds": 0.0,
        "passed": not missing,
    })
    if missing:
        raise RuntimeError(f"README is missing exact quick-start commands: {missing!r}")

    with tempfile.TemporaryDirectory(prefix="worldzero-readme-quickstart-") as text:
        work = Path(text)
        source = work / "source"
        venv = work / ".venv"
        _copy_public(root, source)
        env = os.environ.copy()
        env.update({
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        _run_check(
            checks,
            name="fresh_clean_environment",
            argv=["python", "-m", "venv", "--system-site-packages", ".venv"],
            cwd=work,
            python=Path(sys.executable),
            env=env,
        )
        python = venv / "bin/python"
        _run_check(
            checks,
            name="editable_source_install",
            argv=["python", "-m", "pip", "install", "--no-build-isolation", "-e", ".[test]"],
            cwd=source,
            python=python,
            env=env,
        )
        _run_check(
            checks,
            name="built_in_list",
            argv=["python", "-m", "worldzero", "laws", "list"],
            cwd=source,
            python=python,
            env=env,
            validate=lambda result: _json(result.stdout).get("family_ids") == sorted(FAMILY_IDS),
        )
        for family_id in FAMILY_IDS:
            suffix = family_id.split(":", 1)[1].replace("-", "_")
            _run_check(
                checks,
                name=f"inspect_{suffix}",
                argv=["python", "-m", "worldzero", "laws", "inspect", family_id],
                cwd=source,
                python=python,
                env=env,
                validate=lambda result, expected=family_id: (
                    _json(result.stdout).get("family_id") == expected
                    and _json(result.stdout).get("official") is True
                ),
            )
            _run_check(
                checks,
                name=f"validate_{suffix}",
                argv=["python", "-m", "worldzero", "laws", "validate", family_id, "--seeds", "1"],
                cwd=source,
                python=python,
                env=env,
                validate=lambda result, expected=family_id: (
                    _json(result.stdout).get("family_id") == expected
                    and _json(result.stdout).get("passed") is True
                ),
            )
        _run_check(
            checks,
            name="builtin_demo_capture",
            argv=["python", "-m", "worldzero", "demo", "--seeds", "1", "--output", "worldzero-demo"],
            cwd=source,
            python=python,
            env=env,
        )
        _run_check(
            checks,
            name="builtin_exact_replay",
            argv=["python", "-m", "worldzero", "replay", "worldzero-demo/traces/pressure-experimenter/1452232541.json.gz"],
            cwd=source,
            python=python,
            env=env,
            validate=lambda result: _json(result.stdout).get("verified") is True,
        )
        _run_check(
            checks,
            name="agent_challenge_manifest",
            argv=[
                "python", "-m", "worldzero", "benchmark", "create-manifest",
                "--output", "benchmark.json", "--dev-count", "1", "--test-count", "1",
            ],
            cwd=source,
            python=python,
            env=env,
            validate=lambda result: (
                _json(result.stdout).get("suite", {}).get("suite_id") == "worldzero:core-v1"
            ),
        )
        _run_check(
            checks,
            name="custom_agent_challenge",
            argv=[
                "python", "-m", "worldzero", "benchmark", "run",
                "--manifest", "benchmark.json",
                "--agent", "examples.custom_agent:create_agent",
                "--agent-version", "0.1.0", "--split", "dev", "--no-baselines",
                "--output", "runs/custom-agent",
            ],
            cwd=source,
            python=python,
            env=env,
            validate=lambda result: (
                _json(result.stdout).get("profile", {}).get("coverage", {})
                .get("active") == 3
            ),
        )
        _run_check(
            checks,
            name="example_wheel_build",
            argv=[
                "python", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                "--wheel-dir", "example-dist", "./examples/community_law_plugin",
            ],
            cwd=source,
            python=python,
            env=env,
        )
        wheel = "example-dist/worldzero_example_law-0.1.0-py3-none-any.whl"
        _run_check(
            checks,
            name="example_wheel_install",
            argv=["python", "-m", "pip", "install", "--no-index", "--no-deps", wheel],
            cwd=source,
            python=python,
            env=env,
        )
        _run_check(
            checks,
            name="example_inspect_trust_warning",
            argv=["python", "-m", "worldzero", "laws", "inspect", "example_org:preserver", "--experimental-family"],
            cwd=source,
            python=python,
            env=env,
            validate=lambda result: (
                WARNING in result.stderr
                and _json(result.stdout).get("family_id") == "example_org:preserver"
                and _json(result.stdout).get("experimental") is True
            ),
        )
        _run_check(
            checks,
            name="example_validate_trust_warning",
            argv=[
                "python", "-m", "worldzero", "laws", "validate", "example_org:preserver",
                "--experimental-family", "--seeds", "1",
            ],
            cwd=source,
            python=python,
            env=env,
            validate=lambda result: WARNING in result.stderr and _json(result.stdout).get("passed") is True,
        )
        _run_check(
            checks,
            name="example_official_refusal",
            argv=["python", "-m", "worldzero", "laws", "inspect", "example_org:preserver"],
            cwd=source,
            python=python,
            env=env,
            expected_exit_code=2,
            validate=lambda result: "requires --experimental-family" in result.stderr,
        )
        _run_check(
            checks,
            name="experimental_demo_capture",
            argv=[
                "python", "-m", "worldzero", "demo", "--seeds", "1", "--law-family",
                "example_org:preserver", "--experimental-family", "--output",
                "worldzero-example-demo",
            ],
            cwd=source,
            python=python,
            env=env,
            validate=lambda result: WARNING in result.stderr,
        )
        _run_check(
            checks,
            name="experimental_exact_replay",
            argv=[
                "python", "-m", "worldzero", "replay",
                "worldzero-example-demo/traces/pressure-experimenter/1452232541.json.gz",
            ],
            cwd=source,
            python=python,
            env=env,
            validate=lambda result: _json(result.stdout).get("verified") is True,
        )

    print(json.dumps({
        "schema": "worldzero-readme-quickstart-v1",
        "checks": checks,
        "clean_public_allowlist_copy": True,
        "fresh_virtual_environment": True,
        "pip_no_index": True,
        "external_network_used": False,
        "model_endpoint_used": False,
        "paid_requests": 0,
        "passed": all(row["passed"] for row in checks),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
