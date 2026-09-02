"""Deterministic public-tree and Python-distribution hygiene checks."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

PUBLIC_MANIFEST = Path("docs/public-files.json")
MAX_PUBLIC_BYTES = 1_000_000

# This detached record contains the hashes of the release distributions.  It is
# public source, but cannot be a member of an archive whose own hash it records.
DETACHED_RELEASE_FILES = frozenset({
    "release-verification.json",
    *(f"evidence/release/commands/{command_id}.json" for command_id in (
        "complete_test_suite",
        "math_768",
        "validate_catalysis",
        "validate_inhibition",
        "validate_delayed",
        "validate_null",
        "replay_invariants",
        "workspace_hygiene",
        "build_offline_smoke",
        "readme_quickstart",
    )),
})

IGNORED_ROOTS = frozenset(
    {
        ".git",
        ".local-archive",
        ".release-checkpoints",
        ".superpowers",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
        "runs",
        "pilot_evidence",
        "worldzero-demo",
        "worldzero-validation-reproduction",
        "example-dist",
        "worldzero_research.egg-info",
    }
)

FORBIDDEN_PARTS = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".superpowers",
        ".release-checkpoints",
        ".local-archive",
        ".venv",
        "runs",
        "pilot_evidence",
        "build",
        "dist",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {".pyc", ".pyo", ".db", ".sqlite", ".log", ".sock", ".egg-info"}
)
FORBIDDEN_BASENAMES = frozenset({"r6_pilot.py", "pilot.py"})
ALLOWED_BINARY_SUFFIXES = frozenset({".gz"})

_ABSOLUTE_LOCAL = re.compile(
    "|".join(
        re.escape(prefix)
        for prefix in ("/" + "Users/", "/" + "home/", "C:" + "\\" + "Users" + "\\")
    )
)
_MARKDOWN_LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
_LIVE_SECRET_PATTERNS = (
    ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("openai_key", re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{48,}")),
    ("aws_access_key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{32,}")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_identity(path: Path) -> dict[str, object]:
    """Hash a file or tree over sorted POSIX-relative names and file digests."""
    if path.is_file():
        return {"bytes": path.stat().st_size, "files": 1, "sha256": sha256_file(path)}
    digest = hashlib.sha256()
    size = 0
    count = 0
    for member in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = PurePosixPath(path.name) / PurePosixPath(member.relative_to(path).as_posix())
        member_digest = bytes.fromhex(sha256_file(member))
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(member_digest)
        size += member.stat().st_size
        count += 1
    return {"bytes": size, "files": count, "sha256": digest.hexdigest()}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_schema_identities(root: Path) -> dict[str, object]:
    """Recompute exact state/trace codec identities from final source and fixtures."""
    from . import experiment
    from .kernel import World

    def source_rows(*names: str) -> list[dict[str, str]]:
        return [{"path": name, "sha256": sha256_file(root / name)} for name in names]

    def json_fields(name: str, *, compressed: bool = False) -> list[str]:
        path = root / name
        if compressed:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
        if type(value) is not dict:
            raise ValueError(f"schema fixture is not an object: {name}")
        return sorted(value)

    raw_contracts = [
        {
            "schema_id": "worldzero-state-v2",
            "kind": "state",
            "source_files": source_rows(
                "worldzero/kernel.py",
                "tests/fixtures/legacy/state-v2.json",
                "tests/fixtures/legacy/state-v2-null.json",
            ),
            "field_manifest": {
                "top_fields": json_fields("tests/fixtures/legacy/state-v2.json"),
                "null_top_fields": json_fields("tests/fixtures/legacy/state-v2-null.json"),
            },
        },
        {
            "schema_id": "worldzero-state-v3",
            "kind": "state",
            "source_files": source_rows("worldzero/kernel.py"),
            "field_manifest": {
                "top_fields": sorted(World._plugin_snapshot_fields),
                "family_fields": sorted(World._plugin_family_fields),
                "instance_fields": sorted(World._plugin_instance_fields),
                "agent_fields": sorted(World._plugin_agent_fields),
                "law_fields": sorted(World._plugin_law_fields),
                "config_fields": sorted(World._config_fields),
                "audit_fields": sorted(World._audit_fields),
                "pcg64_fields": sorted(World._pcg64_fields),
                "pcg64_state_fields": sorted(World._pcg64_state_fields),
                "event_kinds": sorted(World._event_kinds),
            },
        },
        {
            "schema_id": "worldzero-trace-v2",
            "kind": "trace",
            "source_files": source_rows(
                "worldzero/experiment.py",
                "tests/fixtures/legacy/trace-v2.json.gz",
                "tests/fixtures/legacy/trace-v2-null.json.gz",
            ),
            "field_manifest": {
                "top_fields": json_fields("tests/fixtures/legacy/trace-v2.json.gz", compressed=True),
                "null_top_fields": json_fields("tests/fixtures/legacy/trace-v2-null.json.gz", compressed=True),
            },
        },
        {
            "schema_id": "worldzero-trace-v3",
            "kind": "trace",
            "source_files": source_rows(
                "worldzero/experiment.py",
                "tests/fixtures/legacy/trace-v3.json.gz",
                "tests/fixtures/legacy/trace-v3-null.json.gz",
            ),
            "field_manifest": {
                "top_fields": json_fields("tests/fixtures/legacy/trace-v3.json.gz", compressed=True),
                "null_top_fields": json_fields("tests/fixtures/legacy/trace-v3-null.json.gz", compressed=True),
            },
        },
        {
            "schema_id": "worldzero-trace-v4",
            "kind": "trace",
            "source_files": source_rows("worldzero/experiment.py", "worldzero/kernel.py"),
            "field_manifest": {
                "top_fields": sorted(experiment._PLUGIN_TRACE_FIELDS),
                "identity_fields": sorted(experiment._PLUGIN_IDENTITY_FIELDS),
                "decision_fields": sorted(experiment._PLUGIN_DECISION_FIELDS),
                "result_fields": sorted(experiment._PLUGIN_RESULT_FIELDS),
                "evaluator_baseline_fields": sorted(experiment._PLUGIN_EVALUATOR_BASELINE_FIELDS),
                "initial_state_schema": World.plugin_schema,
            },
        },
    ]
    contracts: list[dict[str, object]] = []
    for raw in raw_contracts:
        contracts.append({**raw, "identity_sha256": _canonical_sha256(raw)})
    return {"schema": "worldzero-schema-identities-v1", "contracts": contracts}


def write_schema_identities(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(expected_schema_identities(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def candidate_public_files(root: Path) -> tuple[str, ...]:
    candidates: list[str] = []
    sdist_generated = (root / "PKG-INFO").is_file() and any(root.glob("*.egg-info"))
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in IGNORED_ROOTS:
            continue
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if sdist_generated and relative.as_posix() in {"PKG-INFO", "setup.cfg"}:
            continue
        candidates.append(relative.as_posix())
    return tuple(sorted(candidates))


def scan_secret_types(path: Path) -> tuple[str, ...]:
    data = path.read_bytes()
    return tuple(name for name, pattern in _LIVE_SECRET_PATTERNS if pattern.search(data))


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RELEASE_COMMAND_FIELDS = {
    "id", "argv", "cwd", "environment", "parser", "log", "exit_code",
    "passed", "pass_count", "fail_count", "error_count", "duration_seconds",
    "capture_status",
}
_RELEASE_LOG_FIELDS = {
    "schema", "id", "argv", "cwd", "environment", "parser", "exit_code",
    "duration_seconds", "stdout", "stderr",
}
_README_QUICKSTART_COMMANDS = {
    "readme_command_manifest": ["README.md"],
    "fresh_clean_environment": ["python", "-m", "venv", "--system-site-packages", ".venv"],
    "editable_source_install": ["python", "-m", "pip", "install", "--no-build-isolation", "-e", ".[test]"],
    "built_in_list": ["python", "-m", "worldzero", "laws", "list"],
    "inspect_catalysis": ["python", "-m", "worldzero", "laws", "inspect", "worldzero:catalysis"],
    "validate_catalysis": ["python", "-m", "worldzero", "laws", "validate", "worldzero:catalysis", "--seeds", "1"],
    "inspect_inhibition": ["python", "-m", "worldzero", "laws", "inspect", "worldzero:inhibition"],
    "validate_inhibition": ["python", "-m", "worldzero", "laws", "validate", "worldzero:inhibition", "--seeds", "1"],
    "inspect_delayed_transformation": ["python", "-m", "worldzero", "laws", "inspect", "worldzero:delayed-transformation"],
    "validate_delayed_transformation": ["python", "-m", "worldzero", "laws", "validate", "worldzero:delayed-transformation", "--seeds", "1"],
    "inspect_null": ["python", "-m", "worldzero", "laws", "inspect", "worldzero:null"],
    "validate_null": ["python", "-m", "worldzero", "laws", "validate", "worldzero:null", "--seeds", "1"],
    "builtin_demo_capture": ["python", "-m", "worldzero", "demo", "--seeds", "1", "--output", "worldzero-demo"],
    "builtin_exact_replay": ["python", "-m", "worldzero", "replay", "worldzero-demo/traces/pressure-experimenter/1452232541.json.gz"],
    "example_wheel_build": ["python", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", "example-dist", "./examples/community_law_plugin"],
    "example_wheel_install": ["python", "-m", "pip", "install", "--no-index", "--no-deps", "example-dist/worldzero_example_law-0.1.0-py3-none-any.whl"],
    "example_inspect_trust_warning": ["python", "-m", "worldzero", "laws", "inspect", "example_org:preserver", "--experimental-family"],
    "example_validate_trust_warning": ["python", "-m", "worldzero", "laws", "validate", "example_org:preserver", "--experimental-family", "--seeds", "1"],
    "example_official_refusal": ["python", "-m", "worldzero", "laws", "inspect", "example_org:preserver"],
    "experimental_demo_capture": ["python", "-m", "worldzero", "demo", "--seeds", "1", "--law-family", "example_org:preserver", "--experimental-family", "--output", "worldzero-example-demo"],
    "experimental_exact_replay": ["python", "-m", "worldzero", "replay", "worldzero-example-demo/traces/pressure-experimenter/1452232541.json.gz"],
}
_README_SMOKE_GROUPS = {
    "source_install": (
        "readme_command_manifest", "fresh_clean_environment", "editable_source_install",
    ),
    "four_builtin_workflow": (
        "built_in_list", "inspect_catalysis", "validate_catalysis",
        "inspect_inhibition", "validate_inhibition",
        "inspect_delayed_transformation", "validate_delayed_transformation",
        "inspect_null", "validate_null",
    ),
    "demo_replay": ("builtin_demo_capture", "builtin_exact_replay"),
    "example_build_install_validate": (
        "example_wheel_build", "example_wheel_install",
        "example_inspect_trust_warning", "example_validate_trust_warning",
    ),
    "experimental_episode_replay": (
        "experimental_demo_capture", "experimental_exact_replay",
    ),
    "official_refusal": ("example_official_refusal",),
}
_MANDATORY_RELEASE_COMMANDS = {
    "complete_test_suite": (["python", "-m", "pytest", "-q"], "pytest"),
    "math_768": (["python", "-m", "worldzero", "check-math", "--samples", "768"], "math"),
    "validate_catalysis": (["python", "-m", "worldzero", "laws", "validate", "worldzero:catalysis", "--seeds", "1"], "family_validation"),
    "validate_inhibition": (["python", "-m", "worldzero", "laws", "validate", "worldzero:inhibition", "--seeds", "1"], "family_validation"),
    "validate_delayed": (["python", "-m", "worldzero", "laws", "validate", "worldzero:delayed-transformation", "--seeds", "1"], "family_validation"),
    "validate_null": (["python", "-m", "worldzero", "laws", "validate", "worldzero:null", "--seeds", "1"], "family_validation"),
    "replay_invariants": ([
        "python", "-m", "pytest", "-q", "tests/test_legacy_compatibility.py",
        "tests/test_kernel.py", "tests/test_trace_v4.py", "tests/test_protocol.py",
        "tests/test_scoring.py", "tests/test_policies.py", "tests/laws/test_testing_kit.py",
        "tests/laws/test_task6_post_final.py",
    ], "pytest"),
    "workspace_hygiene": (["python", "-m", "worldzero.release_hygiene", "workspace", ".", "--skip-release-record"], "hygiene"),
    "build_offline_smoke": (["python", "scripts/release/build_offline_smoke.py"], "build_offline"),
    "readme_quickstart": (["python", "scripts/release/verify_readme_quickstart.py"], "readme_quickstart"),
}


def _closed(value: object, fields: set[str], path: str, errors: list[str]) -> dict[str, object] | None:
    if type(value) is not dict:
        errors.append(f"{path} must be an object")
        return None
    if set(value) != fields:
        errors.append(
            f"{path} fields are not closed: missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )
        return None
    return value


def _digest_value(value: object, path: str, errors: list[str]) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{path} must be a lowercase SHA-256")


def _string(value: object, path: str, errors: list[str], *, choices: set[str] | None = None) -> None:
    if type(value) is not str or not value:
        errors.append(f"{path} must be a non-empty string")
    elif choices is not None and value not in choices:
        errors.append(f"{path} has an unsupported value")


def _boolean(value: object, path: str, errors: list[str]) -> None:
    if type(value) is not bool:
        errors.append(f"{path} must be boolean")


def _count(value: object, path: str, errors: list[str]) -> None:
    if type(value) is not int or value < 0:
        errors.append(f"{path} must be a non-negative integer")


def _duration(value: object, path: str, errors: list[str]) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        errors.append(f"{path} must be a finite non-negative number")


def _release_root(path: Path) -> Path | None:
    for candidate in (path.parent, Path.cwd()):
        if (candidate / "docs/public-files.json").is_file():
            return candidate.resolve()
    return None


def _parse_release_output(parser: str, stdout: str, errors: list[str], path: str) -> tuple[int, int, int] | None:
    if parser == "pytest":
        match = re.search(r"(?m)(\d+) passed(?:, (\d+) failed)?(?:, (\d+) errors?)? in [0-9.]+s", stdout)
        if not match:
            errors.append(f"{path} does not contain a pytest summary")
            return None
        return int(match.group(1)), int(match.group(2) or 0), int(match.group(3) or 0)
    if parser in {"math", "family_validation", "build_offline", "readme_quickstart"}:
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            errors.append(f"{path} stdout is not JSON")
            return None
        if type(result) is not dict:
            errors.append(f"{path} stdout JSON must be an object")
            return None
        if parser == "math":
            expected_fields = {
                "aggregate", "checks", "entropy_example", "families", "passed",
                "samples_per_check", "tolerance",
            }
            if set(result) != expected_fields:
                errors.append(f"{path} math result fields are not closed")
                return None
            families = result.get("families")
            if (result.get("passed") is not True or result.get("samples_per_check") != 768
                    or type(families) is not list or len(families) != 4
                    or any(type(row) is not dict or row.get("passed") is not True
                           or row.get("failures") != [] for row in families)):
                errors.append(f"{path} math result is not passing")
                return None
            count = len(result.get("checks", [])) + sum(len(row.get("cases", [])) for row in families if type(row) is dict)
            return count, 0, 0
        if parser == "family_validation":
            expected_fields = {
                "calibration_suite_sha256", "checks", "descriptor", "experimental",
                "failures", "family_id", "fingerprint", "official", "origin",
                "passed", "schema", "seed_count",
            }
            if set(result) != expected_fields:
                errors.append(f"{path} family validation fields are not closed")
                return None
            checks = result.get("checks")
            if (result.get("schema") != "worldzero-family-validation-v1"
                    or result.get("passed") is not True or result.get("failures") != []
                    or result.get("seed_count") != 1 or result.get("official") is not True
                    or result.get("experimental") is not False or type(checks) is not list
                    or len(checks) != 11
                    or not all(type(row) is dict and set(row) == {"failures", "name", "passed"}
                               and row.get("passed") is True and row.get("failures") == []
                               for row in checks)):
                errors.append(f"{path} family validation is not passing")
                return None
            return len(checks), 0, 0
        if parser == "readme_quickstart":
            expected_fields = {
                "schema", "checks", "clean_public_allowlist_copy",
                "fresh_virtual_environment", "pip_no_index", "external_network_used",
                "model_endpoint_used", "paid_requests", "passed",
            }
            checks = result.get("checks")
            if (set(result) != expected_fields
                    or result.get("schema") != "worldzero-readme-quickstart-v1"
                    or result.get("clean_public_allowlist_copy") is not True
                    or result.get("fresh_virtual_environment") is not True
                    or result.get("pip_no_index") is not True
                    or result.get("external_network_used") is not False
                    or result.get("model_endpoint_used") is not False
                    or result.get("paid_requests") != 0
                    or result.get("passed") is not True
                    or type(checks) is not list
                    or len(checks) != len(_README_QUICKSTART_COMMANDS)):
                errors.append(f"{path} README quick-start result is not closed and passing")
                return None
            names: list[str] = []
            for index, check in enumerate(checks):
                if type(check) is not dict or set(check) != {
                    "name", "argv", "exit_code", "expected_exit_code",
                    "duration_seconds", "passed",
                }:
                    errors.append(f"{path} README check {index} fields are not closed")
                    return None
                name = check.get("name")
                names.append(name if type(name) is str else "")
                expected_exit = 2 if name == "example_official_refusal" else 0
                if (name not in _README_QUICKSTART_COMMANDS
                        or check.get("argv") != _README_QUICKSTART_COMMANDS.get(name)
                        or check.get("expected_exit_code") != expected_exit
                        or check.get("exit_code") != expected_exit
                        or check.get("passed") is not True):
                    errors.append(f"{path} README check {index} contradicts its exact command")
                    return None
                _duration(check.get("duration_seconds"), f"{path}.checks[{index}].duration_seconds", errors)
            if names != list(_README_QUICKSTART_COMMANDS):
                errors.append(f"{path} README check order/names are not exact")
                return None
            return len(checks), 0, 0
        checks = result.get("checks")
        expected_build_fields = {
            "artifacts", "checks", "example_validation", "experimental_replay",
            "external_network_used", "family_ids", "model_endpoint_used",
            "official_refusal", "offline_install", "schema", "source_date_epoch",
        }
        artifacts = result.get("artifacts")
        replay = result.get("experimental_replay")
        if (set(result) != expected_build_fields
                or result.get("schema") != "worldzero-build-offline-smoke-v1"
                or result.get("source_date_epoch") != 1704067200
                or result.get("offline_install") is not True
                or result.get("example_validation") is not True
                or result.get("official_refusal") is not True
                or result.get("external_network_used") is not False
                or result.get("model_endpoint_used") is not False
                or type(artifacts) is not list or len(artifacts) != 3
                or any(type(row) is not dict or set(row) != {
                    "container_bytes_equal", "member_bytes_equal", "member_count",
                    "member_manifest_sha256", "name", "second_sha256", "sha256",
                } for row in artifacts)
                or type(replay) is not dict or set(replay) != {"decisions", "history_sha256", "verified"}
                or replay.get("verified") is not True
                or type(checks) is not list or not checks
                or not all(type(row) is str for row in checks)):
            errors.append(f"{path} build smoke lacks a closed passing check list")
            return None
        return len(checks), 0, 0
    if parser == "hygiene":
        if stdout.strip() != "hygiene: passed":
            errors.append(f"{path} hygiene output is not passing")
            return None
        return 1, 0, 0
    errors.append(f"{path} has unsupported parser")
    return None


def _readme_smoke_from_result(result: dict[str, object]) -> dict[str, object]:
    checks = result.get("checks")
    rows = checks if type(checks) is list else []
    passed_names = {
        row["name"] for row in rows
        if type(row) is dict and row.get("passed") is True
        and type(row.get("name")) is str
    }
    values = {
        field: all(name in passed_names for name in required)
        for field, required in _README_SMOKE_GROUPS.items()
    }
    return {"status": "passed" if all(values.values()) else "failed", **values}


def check_release_verification(path: Path) -> list[str]:
    """Validate and cross-check the detached release record recursively."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return [f"invalid release verification JSON: {exc}"]
    if type(payload) is not dict:
        return ["release verification must be a JSON object"]
    required = {
        "schema",
        "release",
        "generated_at_utc",
        "environment",
        "checkpoints",
        "drift",
        "commands",
        "identities",
        "schemas",
        "compatibility",
        "distributions",
        "cleanup",
        "hygiene",
        "readme_smoke",
        "reviewer",
        "open_source_ready",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        return [f"release verification fields are not closed: missing={missing}, extra={extra}"]
    errors: list[str] = []
    if payload["schema"] != "worldzero-release-verification-v1":
        errors.append("unsupported release verification schema")
    _string(payload["generated_at_utc"], "generated_at_utc", errors)
    if type(payload["generated_at_utc"]) is str and re.fullmatch(
        r"20\d\d-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\dZ",
        payload["generated_at_utc"],
    ) is None:
        errors.append("generated_at_utc must be a closed UTC timestamp")
    _boolean(payload["open_source_ready"], "open_source_ready", errors)

    release = _closed(payload["release"], {"package", "version", "law_api_version", "release_id", "status"}, "release", errors)
    if release:
        for name in ("package", "version", "law_api_version", "release_id"):
            _string(release[name], f"release.{name}", errors)
        expected_release = {
            "package": "worldzero-research",
            "version": "0.3.0",
            "law_api_version": "1.0",
            "release_id": "0.3.0-20260901",
        }
        for name, expected in expected_release.items():
            if release[name] != expected:
                errors.append(f"release.{name} disagrees with the frozen release")
        _string(release["status"], "release.status", errors, choices={
            "verification_capture_in_progress",
            "verification_complete_pending_independent_review", "release_ready",
        })

    environment = _closed(payload["environment"], {
        "platform", "python", "dependencies", "external_network_used",
        "model_endpoint_used", "paid_requests",
    }, "environment", errors)
    if environment:
        _string(environment["platform"], "environment.platform", errors)
        _string(environment["python"], "environment.python", errors)
        dependencies = _closed(environment["dependencies"], {"build", "numpy", "pytest", "setuptools", "wheel"}, "environment.dependencies", errors)
        if dependencies:
            for name, value in dependencies.items():
                _string(value, f"environment.dependencies.{name}", errors)
        _boolean(environment["external_network_used"], "environment.external_network_used", errors)
        _boolean(environment["model_endpoint_used"], "environment.model_endpoint_used", errors)
        _count(environment["paid_requests"], "environment.paid_requests", errors)
        if environment["external_network_used"] is not False:
            errors.append("release verification must not use external network")
        if environment["model_endpoint_used"] is not False:
            errors.append("release verification must not use a model endpoint")
        if environment["paid_requests"] != 0:
            errors.append("release verification paid_requests must be zero")

    checkpoint_fields = {*(f"task_{number}_sha256" for number in range(1, 8)), "task_5_final_review_sha256", "task_6_final_review_finding_sha256", "task_7_review_sha256"}
    checkpoints = _closed(payload["checkpoints"], checkpoint_fields, "checkpoints", errors)
    if checkpoints:
        for name, value in checkpoints.items():
            _digest_value(value, f"checkpoints.{name}", errors)

    drift = _closed(payload["drift"], {"status", "rows", "unexplained_rows"}, "drift", errors)
    if drift:
        _string(drift["status"], "drift.status", errors, choices={"explained"})
        if type(drift["rows"]) is not list or not drift["rows"]:
            errors.append("drift.rows must be a non-empty list")
        else:
            for index, row_value in enumerate(drift["rows"]):
                row = _closed(row_value, {"from", "to", "reason"}, f"drift.rows[{index}]", errors)
                if row:
                    for name in row:
                        _string(row[name], f"drift.rows[{index}].{name}", errors)
        if drift["unexplained_rows"] != []:
            errors.append("drift.unexplained_rows must be empty")

    commands = payload["commands"]
    command_rows: dict[str, dict[str, object]] = {}
    root = _release_root(path)
    if type(commands) is not list or not commands:
        errors.append("commands must be a non-empty list")
    else:
        for index, row_value in enumerate(commands):
            row = _closed(row_value, _RELEASE_COMMAND_FIELDS, f"commands[{index}]", errors)
            if not row:
                continue
            command_id = row["id"]
            _string(command_id, f"commands[{index}].id", errors)
            if type(command_id) is str:
                if command_id in command_rows:
                    errors.append("command ids must be unique")
                command_rows[command_id] = row
            if type(row["argv"]) is not list or not row["argv"] or not all(type(value) is str and value for value in row["argv"]):
                errors.append(f"commands[{index}].argv must be a non-empty string list")
            if row["cwd"] != ".":
                errors.append(f"commands[{index}].cwd must be repository root '.'")
            if type(row["environment"]) is not dict or not all(type(key) is str and type(value) is str for key, value in row["environment"].items()):
                errors.append(f"commands[{index}].environment must be a string mapping")
            _string(row["parser"], f"commands[{index}].parser", errors, choices={"pytest", "math", "family_validation", "hygiene", "build_offline", "readme_quickstart"})
            _string(row["capture_status"], f"commands[{index}].capture_status", errors, choices={"capture_pending", "captured"})
            _boolean(row["passed"], f"commands[{index}].passed", errors)
            pending = row["capture_status"] == "capture_pending"
            if pending:
                if command_id != "complete_test_suite":
                    errors.append(f"commands[{index}] only the complete suite may be capture_pending")
                if row["log"] is not None or any(row[name] is not None for name in (
                    "exit_code", "pass_count", "fail_count", "error_count", "duration_seconds"
                )) or row["passed"] is not False:
                    errors.append(f"commands[{index}] capture_pending values conflict")
            else:
                for name in ("exit_code", "pass_count", "fail_count", "error_count"):
                    _count(row[name], f"commands[{index}].{name}", errors)
                _duration(row["duration_seconds"], f"commands[{index}].duration_seconds", errors)
                success = row["exit_code"] == 0 and row["fail_count"] == 0 and row["error_count"] == 0 and row["pass_count"] > 0
                if row["passed"] is not success:
                    errors.append(f"commands[{index}] pass/exit/count values conflict")
            log = None if pending else _closed(row["log"], {"path", "sha256", "bytes"}, f"commands[{index}].log", errors)
            if log:
                _string(log["path"], f"commands[{index}].log.path", errors)
                _digest_value(log["sha256"], f"commands[{index}].log.sha256", errors)
                _count(log["bytes"], f"commands[{index}].log.bytes", errors)
                expected_log = f"evidence/release/commands/{command_id}.json"
                if log["path"] != expected_log:
                    errors.append(f"commands[{index}] log path is not exact")
                if root and type(log["path"]) is str:
                    log_path = root / log["path"]
                    if not log_path.is_file():
                        errors.append(f"commands[{index}] log is missing")
                    else:
                        if sha256_file(log_path) != log["sha256"] or log_path.stat().st_size != log["bytes"]:
                            errors.append(f"commands[{index}] log identity mismatch")
                        try:
                            log_payload = json.loads(log_path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                        except (json.JSONDecodeError, UnicodeError, ValueError):
                            errors.append(f"commands[{index}] log JSON is invalid")
                        else:
                            closed_log = _closed(log_payload, _RELEASE_LOG_FIELDS, f"commands[{index}].log_payload", errors)
                            if closed_log:
                                if closed_log["schema"] != "worldzero-release-command-log-v1":
                                    errors.append(f"commands[{index}] log schema is unsupported")
                                for name in ("id", "argv", "cwd", "environment", "parser", "exit_code", "duration_seconds"):
                                    if closed_log[name] != row[name]:
                                        errors.append(f"commands[{index}] log disagrees on {name}")
                                if type(closed_log["stdout"]) is not str or type(closed_log["stderr"]) is not str:
                                    errors.append(f"commands[{index}] log streams must be strings")
                                    continue
                                local_prefixes = (
                                    "/" + "Users/",
                                    "/" + "home/",
                                    "C:" + "\\" + "Users" + "\\",
                                )
                                if any(prefix in (closed_log["stdout"] + closed_log["stderr"]) for prefix in local_prefixes):
                                    errors.append(f"commands[{index}] log exposes an absolute developer path")
                                parsed = _parse_release_output(row["parser"], closed_log["stdout"], errors, f"commands[{index}]")
                                if parsed and parsed != (row["pass_count"], row["fail_count"], row["error_count"]):
                                    errors.append(f"commands[{index}] parsed counts disagree with record")
                                expected_validation_ids = {
                                    "validate_catalysis": "worldzero:catalysis",
                                    "validate_inhibition": "worldzero:inhibition",
                                    "validate_delayed": "worldzero:delayed-transformation",
                                    "validate_null": "worldzero:null",
                                }
                                if command_id in expected_validation_ids:
                                    try:
                                        validation = json.loads(closed_log["stdout"])
                                    except json.JSONDecodeError:
                                        pass
                                    else:
                                        if validation.get("family_id") != expected_validation_ids[command_id]:
                                            errors.append(f"commands[{index}] validates the wrong family")
    if set(command_rows) != set(_MANDATORY_RELEASE_COMMANDS):
        errors.append("commands do not contain the exact mandatory gate ids")
    for command_id, (argv, parser_name) in _MANDATORY_RELEASE_COMMANDS.items():
        row = command_rows.get(command_id)
        if row and (row["argv"] != argv or row["parser"] != parser_name or row["environment"] != {}):
            errors.append(f"command {command_id} is not the exact mandatory executable command")

    identities = _closed(payload["identities"], {"official_registry_sha256", "scoring_profile", "official_families"}, "identities", errors)
    if identities:
        _digest_value(identities["official_registry_sha256"], "identities.official_registry_sha256", errors)
        profile = _closed(identities["scoring_profile"], {"profile_id", "version", "thresholds_sha256"}, "identities.scoring_profile", errors)
        if profile:
            _string(profile["profile_id"], "identities.scoring_profile.profile_id", errors)
            _string(profile["version"], "identities.scoring_profile.version", errors)
            _digest_value(profile["thresholds_sha256"], "identities.scoring_profile.thresholds_sha256", errors)
        families = identities["official_families"]
        expected_family_ids = {"worldzero:catalysis", "worldzero:delayed-transformation", "worldzero:inhibition", "worldzero:null"}
        seen_family_ids: set[str] = set()
        if type(families) is not list or len(families) != 4:
            errors.append("identities.official_families must contain exactly four rows")
        else:
            for index, family_value in enumerate(families):
                family = _closed(family_value, {"family_id", "family_version", "fingerprint", "calibration_suite_sha256", "validation_passed"}, f"identities.official_families[{index}]", errors)
                if family:
                    _string(family["family_id"], f"identities.official_families[{index}].family_id", errors)
                    _string(family["family_version"], f"identities.official_families[{index}].family_version", errors)
                    _digest_value(family["fingerprint"], f"identities.official_families[{index}].fingerprint", errors)
                    _digest_value(family["calibration_suite_sha256"], f"identities.official_families[{index}].calibration_suite_sha256", errors)
                    _boolean(family["validation_passed"], f"identities.official_families[{index}].validation_passed", errors)
                    if type(family["family_id"]) is str:
                        seen_family_ids.add(family["family_id"])
        if seen_family_ids != expected_family_ids:
            errors.append("official family ids are not exact")
        if any(type(row) is dict and row.get("validation_passed") is not True for row in families if type(families) is list):
            errors.append("every official family validation must pass")

    schemas = _closed(payload["schemas"], {"manifest", "contracts"}, "schemas", errors)
    if schemas:
        manifest = _closed(schemas["manifest"], {"path", "sha256", "bytes"}, "schemas.manifest", errors)
        if manifest:
            _string(manifest["path"], "schemas.manifest.path", errors)
            _digest_value(manifest["sha256"], "schemas.manifest.sha256", errors)
            _count(manifest["bytes"], "schemas.manifest.bytes", errors)
            if manifest["path"] != "evidence/release/schema-identities.json":
                errors.append("schemas.manifest.path is not exact")
        contracts = schemas["contracts"]
        if type(contracts) is not list or len(contracts) != 5:
            errors.append("schemas.contracts must contain exactly five rows")
        else:
            expected_ids = {"worldzero-state-v2", "worldzero-state-v3", "worldzero-trace-v2", "worldzero-trace-v3", "worldzero-trace-v4"}
            seen_ids = set()
            for index, contract_value in enumerate(contracts):
                contract = _closed(contract_value, {"schema_id", "identity_sha256"}, f"schemas.contracts[{index}]", errors)
                if contract:
                    _string(contract["schema_id"], f"schemas.contracts[{index}].schema_id", errors)
                    _digest_value(contract["identity_sha256"], f"schemas.contracts[{index}].identity_sha256", errors)
                    if type(contract["schema_id"]) is str:
                        seen_ids.add(contract["schema_id"])
            if seen_ids != expected_ids:
                errors.append("schema contract ids are not exact")

    compatibility = _closed(payload["compatibility"], {
        "legacy_fixture_count", "legacy_replay_status", "trace_v4",
        "invariant_matrix_status", "invariant_matrix", "mixed_key_collision_status",
    }, "compatibility", errors)
    if compatibility:
        _count(compatibility["legacy_fixture_count"], "compatibility.legacy_fixture_count", errors)
        if compatibility["legacy_fixture_count"] != 7:
            errors.append("compatibility.legacy_fixture_count must be seven")
        _string(compatibility["legacy_replay_status"], "compatibility.legacy_replay_status", errors, choices={"passed_7_of_7"})
        _string(compatibility["invariant_matrix_status"], "compatibility.invariant_matrix_status", errors, choices={"passed_in_complete_test_matrix"})
        _string(compatibility["mixed_key_collision_status"], "compatibility.mixed_key_collision_status", errors, choices={"passed_7_of_7"})
        expected_invariants = [
            "forbidden_observation", "detached_mutation", "invalid_transition_atomicity",
            "accounting", "control_matching", "absorbing_death",
            "empty_successor_memory", "administrative_censoring",
            "all_ancestor_effects", "eligible_only_effects",
        ]
        if compatibility["invariant_matrix"] != expected_invariants:
            errors.append("compatibility.invariant_matrix is not the exact ten-check matrix")
        traces = compatibility["trace_v4"]
        if type(traces) is not list or len(traces) != 4:
            errors.append("compatibility.trace_v4 must contain exactly four rows")
        else:
            trace_family_ids: set[str] = set()
            for index, trace_value in enumerate(traces):
                trace = _closed(trace_value, {"family_id", "trace_sha256", "verified"}, f"compatibility.trace_v4[{index}]", errors)
                if trace:
                    _string(trace["family_id"], f"compatibility.trace_v4[{index}].family_id", errors)
                    _digest_value(trace["trace_sha256"], f"compatibility.trace_v4[{index}].trace_sha256", errors)
                    _boolean(trace["verified"], f"compatibility.trace_v4[{index}].verified", errors)
                    if type(trace["family_id"]) is str:
                        trace_family_ids.add(trace["family_id"])
                    if trace["verified"] is not True:
                        errors.append(f"compatibility.trace_v4[{index}] must be verified")
            if trace_family_ids != {"worldzero:catalysis", "worldzero:delayed-transformation", "worldzero:inhibition", "worldzero:null"}:
                errors.append("compatibility.trace_v4 family ids are not exact")

    distributions = _closed(payload["distributions"], {"source_date_epoch", "status", "artifacts", "two_build_comparison"}, "distributions", errors)
    if distributions:
        _count(distributions["source_date_epoch"], "distributions.source_date_epoch", errors)
        if distributions["source_date_epoch"] != 1704067200:
            errors.append("distributions.source_date_epoch is not frozen")
        _string(distributions["status"], "distributions.status", errors, choices={"passed"})
        artifacts = distributions["artifacts"]
        if type(artifacts) is not list or len(artifacts) != 3:
            errors.append("distributions.artifacts must contain exactly three rows")
        else:
            artifact_names: set[str] = set()
            for index, artifact_value in enumerate(artifacts):
                artifact = _closed(artifact_value, {"name", "sha256", "member_count", "member_manifest_sha256"}, f"distributions.artifacts[{index}]", errors)
                if artifact:
                    _string(artifact["name"], f"distributions.artifacts[{index}].name", errors)
                    _digest_value(artifact["sha256"], f"distributions.artifacts[{index}].sha256", errors)
                    _count(artifact["member_count"], f"distributions.artifacts[{index}].member_count", errors)
                    _digest_value(artifact["member_manifest_sha256"], f"distributions.artifacts[{index}].member_manifest_sha256", errors)
                    if type(artifact["name"]) is str:
                        artifact_names.add(artifact["name"])
            if artifact_names != {
                "worldzero_research-0.3.0-py3-none-any.whl",
                "worldzero_example_law-0.1.0-py3-none-any.whl",
                "worldzero_research-0.3.0.tar.gz",
            }:
                errors.append("distribution artifact names are not exact")
        comparison = _closed(distributions["two_build_comparison"], {"worldzero_wheel_bytes_equal", "example_wheel_bytes_equal", "sdist_member_names_and_bytes_equal", "sdist_container_bytes_equal", "sdist_second_sha256", "sdist_container_note"}, "distributions.two_build_comparison", errors)
        if comparison:
            for name in ("worldzero_wheel_bytes_equal", "example_wheel_bytes_equal", "sdist_member_names_and_bytes_equal", "sdist_container_bytes_equal"):
                _boolean(comparison[name], f"distributions.two_build_comparison.{name}", errors)
            if comparison["worldzero_wheel_bytes_equal"] is not True or comparison["example_wheel_bytes_equal"] is not True or comparison["sdist_member_names_and_bytes_equal"] is not True:
                errors.append("distribution reproducibility truth values conflict")
            _digest_value(comparison["sdist_second_sha256"], "distributions.two_build_comparison.sdist_second_sha256", errors)
            _string(comparison["sdist_container_note"], "distributions.two_build_comparison.sdist_container_note", errors)

    cleanup = _closed(payload["cleanup"], {"public_manifest_sha256", "public_file_count", "archive_manifest_sha256", "archive_source_identities_sha256", "recovery_path", "recovery_policy"}, "cleanup", errors)
    if cleanup:
        for name in ("public_manifest_sha256", "archive_manifest_sha256", "archive_source_identities_sha256"):
            _digest_value(cleanup[name], f"cleanup.{name}", errors)
        _count(cleanup["public_file_count"], "cleanup.public_file_count", errors)
        _string(cleanup["recovery_path"], "cleanup.recovery_path", errors)
        _string(cleanup["recovery_policy"], "cleanup.recovery_policy", errors)
        if cleanup["recovery_path"] != ".local-archive/worldzero-pre-open-source-0.3.0-20260901":
            errors.append("cleanup.recovery_path is not exact")

    hygiene = _closed(payload["hygiene"], {"workspace", "credential_patterns", "absolute_developer_paths", "broken_relative_links", "unexpected_large_or_binary_files", "internal_files_in_public_manifest", "git_initialized"}, "hygiene", errors)
    if hygiene:
        _string(hygiene["workspace"], "hygiene.workspace", errors, choices={"passed"})
        for name in ("credential_patterns", "absolute_developer_paths", "broken_relative_links", "unexpected_large_or_binary_files", "internal_files_in_public_manifest"):
            _count(hygiene[name], f"hygiene.{name}", errors)
            if hygiene[name] != 0:
                errors.append(f"hygiene.{name} must be zero")
        _boolean(hygiene["git_initialized"], "hygiene.git_initialized", errors)
        if hygiene["git_initialized"] is not False:
            errors.append("hygiene.git_initialized must be false before repository creation")

    readme = _closed(payload["readme_smoke"], {"status", "source_install", "four_builtin_workflow", "demo_replay", "example_build_install_validate", "experimental_episode_replay", "official_refusal"}, "readme_smoke", errors)
    if readme:
        _string(readme["status"], "readme_smoke.status", errors, choices={"passed"})
        for name in set(readme) - {"status"}:
            _boolean(readme[name], f"readme_smoke.{name}", errors)
            if readme[name] is not True:
                errors.append(f"readme_smoke.{name} must be true")

    reviewer = payload["reviewer"]
    reviewer_fields = {"status", "spec", "quality", "critical", "important", "minor", "report_sha256"}
    reviewer = _closed(reviewer, reviewer_fields, "reviewer", errors)
    reviewer_passed = False
    if reviewer:
        _string(reviewer["status"], "reviewer.status", errors, choices={"pending", "complete"})
        _string(reviewer["spec"], "reviewer.spec", errors, choices={"PENDING", "PASS", "FAIL"})
        _string(reviewer["quality"], "reviewer.quality", errors, choices={"PENDING", "PASS", "FAIL"})
        for name in ("critical", "important", "minor"):
            _count(reviewer[name], f"reviewer.{name}", errors)
        if reviewer["report_sha256"] is not None:
            _digest_value(reviewer["report_sha256"], "reviewer.report_sha256", errors)
        pending_consistent = reviewer["status"] == "pending" and reviewer["spec"] == reviewer["quality"] == "PENDING" and reviewer["report_sha256"] is None
        complete_consistent = reviewer["status"] == "complete" and reviewer["spec"] in {"PASS", "FAIL"} and reviewer["quality"] in {"PASS", "FAIL"} and reviewer["report_sha256"] is not None
        if not (pending_consistent or complete_consistent):
            errors.append("reviewer status/verdict/report fields conflict")
        reviewer_passed = complete_consistent and reviewer["spec"] == reviewer["quality"] == "PASS" and reviewer["critical"] == reviewer["important"] == 0

    pending_commands = [row for row in command_rows.values() if row.get("capture_status") == "capture_pending"]
    all_commands_passed = set(command_rows) == set(_MANDATORY_RELEASE_COMMANDS) and all(
        row.get("capture_status") == "captured" and row.get("passed") is True
        for row in command_rows.values()
    )
    gates_passed = all_commands_passed and drift is not None and drift.get("unexplained_rows") == [] and hygiene is not None and hygiene.get("workspace") == "passed" and distributions is not None and distributions.get("status") == "passed" and readme is not None and readme.get("status") == "passed"
    if payload["open_source_ready"] is True and not (gates_passed and reviewer_passed):
        errors.append("open_source_ready requires every gate and a completed PASS/PASS review")
    if gates_passed and reviewer_passed and payload["open_source_ready"] is not True:
        errors.append("a fully passing reviewed record must set open_source_ready true")
    capture_in_progress = release is not None and release.get("status") == "verification_capture_in_progress"
    if pending_commands:
        if not (capture_in_progress and len(pending_commands) == 1 and reviewer is not None
                and reviewer.get("status") == "pending" and payload["open_source_ready"] is False):
            errors.append("capture_pending requires the unique two-phase pending-review state")
    elif capture_in_progress:
        errors.append("verification_capture_in_progress requires one capture_pending command")
    if release is not None:
        if release.get("status") == "release_ready" and not payload["open_source_ready"]:
            errors.append("release_ready status requires open_source_ready true")
        if release.get("status") == "verification_complete_pending_independent_review" and (
            pending_commands or reviewer is None or reviewer.get("status") != "pending"
        ):
            errors.append("verification_complete_pending_independent_review state conflicts")

    if root:
        # Recompute checkpoint and public/archive identities from their trust roots.
        checkpoint_paths = {**{f"task_{number}_sha256": f".release-checkpoints/task-{number}.json" for number in range(1, 8)},
            "task_5_final_review_sha256": ".superpowers/sdd/2026-08-31-worldzero-law-family-plugins/task-5-final-review.md",
            "task_6_final_review_finding_sha256": ".superpowers/sdd/2026-08-31-worldzero-law-family-plugins/task-6-final-review.md",
            "task_7_review_sha256": ".superpowers/sdd/2026-08-31-worldzero-law-family-plugins/task-7-review.md"}
        if checkpoints:
            checkpoint_files = {
                name: root / relative for name, relative in checkpoint_paths.items()
            }
            if any(path.is_file() for path in checkpoint_files.values()):
                for name, checkpoint_path in checkpoint_files.items():
                    if not checkpoint_path.is_file():
                        errors.append(f"checkpoints.{name} private trust root is missing")
                    elif sha256_file(checkpoint_path) != checkpoints[name]:
                        errors.append(f"checkpoints.{name} disagrees with current source")
        if cleanup:
            public_manifest = root / "docs/public-files.json"
            public_payload = json.loads(public_manifest.read_text(encoding="utf-8"))
            expected_cleanup = {
                "public_manifest_sha256": sha256_file(public_manifest),
                "public_file_count": len(public_payload["files"]),
            }
            archive_root = root / ".local-archive/worldzero-pre-open-source-0.3.0-20260901"
            archive_files = {
                "archive_manifest_sha256": archive_root / "archive-manifest.json",
                "archive_source_identities_sha256": archive_root / "archive-source-identities.json",
            }
            if any(path.is_file() for path in archive_files.values()):
                for name, archive_path in archive_files.items():
                    if not archive_path.is_file():
                        errors.append(f"cleanup.{name} private trust root is missing")
                    else:
                        expected_cleanup[name] = sha256_file(archive_path)
            for name, expected in expected_cleanup.items():
                if cleanup[name] != expected:
                    errors.append(f"cleanup.{name} disagrees with current manifest")
        if identities:
            from .scoring import default_scoring_profile

            if identities["scoring_profile"] != default_scoring_profile().identity_dict():
                errors.append("scoring profile identity disagrees with current source")
            registry_path = root / "worldzero/laws/official_registry.json"
            if sha256_file(registry_path) != identities["official_registry_sha256"]:
                errors.append("official registry identity disagrees with current source")
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry_rows = {row["family_id"]: row for row in registry["approved_families"]}
            for family in identities["official_families"] if type(identities["official_families"]) is list else []:
                if type(family) is dict and family.get("family_id") in registry_rows:
                    approved = registry_rows[family["family_id"]]
                    for name in ("family_version", "fingerprint", "calibration_suite_sha256"):
                        if family.get(name) != approved[name]:
                            errors.append(f"official family {family.get('family_id')} {name} disagrees with registry")
        if schemas:
            manifest = schemas.get("manifest")
            if type(manifest) is dict and type(manifest.get("path")) is str:
                manifest_path = root / manifest["path"]
                if not manifest_path.is_file() or sha256_file(manifest_path) != manifest.get("sha256") or manifest_path.stat().st_size != manifest.get("bytes"):
                    errors.append("schema identity manifest file disagrees with record")
                else:
                    actual_schema_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    expected_schema_payload = expected_schema_identities(root)
                    if actual_schema_payload != expected_schema_payload:
                        errors.append("schema identity manifest disagrees with final codecs/fixtures")
                    contract_rows = [{"schema_id": row["schema_id"], "identity_sha256": row["identity_sha256"]} for row in actual_schema_payload["contracts"]]
                    if schemas.get("contracts") != contract_rows:
                        errors.append("schema contract identities disagree with manifest")
        build_row = command_rows.get("build_offline_smoke")
        if build_row and type(build_row.get("log")) is dict:
            build_log_path = root / build_row["log"]["path"]
            if build_log_path.is_file() and distributions:
                build_log = json.loads(build_log_path.read_text(encoding="utf-8"))
                try:
                    build_result = json.loads(build_log["stdout"])
                except (KeyError, json.JSONDecodeError):
                    pass
                else:
                    summarized = [{name: row[name] for name in ("name", "sha256", "member_count", "member_manifest_sha256")} for row in build_result.get("artifacts", [])]
                    if distributions.get("artifacts") != summarized:
                        errors.append("distribution artifacts disagree with build-smoke log")
                    by_name = {
                        row.get("name"): row for row in build_result.get("artifacts", [])
                        if type(row) is dict
                    }
                    try:
                        sdist = by_name["worldzero_research-0.3.0.tar.gz"]
                        expected_comparison = {
                            "worldzero_wheel_bytes_equal": by_name["worldzero_research-0.3.0-py3-none-any.whl"]["container_bytes_equal"],
                            "example_wheel_bytes_equal": by_name["worldzero_example_law-0.1.0-py3-none-any.whl"]["container_bytes_equal"],
                            "sdist_member_names_and_bytes_equal": sdist["member_bytes_equal"],
                            "sdist_container_bytes_equal": sdist["container_bytes_equal"],
                            "sdist_second_sha256": sdist["second_sha256"],
                        }
                    except (KeyError, TypeError):
                        errors.append("build-smoke artifact comparison output is malformed")
                    else:
                        comparison = distributions.get("two_build_comparison")
                        if type(comparison) is dict and any(
                            comparison.get(name) != value
                            for name, value in expected_comparison.items()
                        ):
                            errors.append("distribution comparison disagrees with build-smoke log")
                    if distributions.get("source_date_epoch") != build_result.get("source_date_epoch"):
                        errors.append("distribution epoch disagrees with build-smoke log")
        readme_row = command_rows.get("readme_quickstart")
        if readme_row and type(readme_row.get("log")) is dict and readme:
            readme_log_path = root / readme_row["log"]["path"]
            if readme_log_path.is_file():
                try:
                    readme_log = json.loads(readme_log_path.read_text(encoding="utf-8"))
                    readme_result = json.loads(readme_log["stdout"])
                except (KeyError, json.JSONDecodeError, UnicodeError):
                    pass
                else:
                    if readme != _readme_smoke_from_result(readme_result):
                        errors.append("readme_smoke disagrees with retained named checks")
    return sorted(set(errors))


def _check_path(relative: PurePosixPath) -> list[str]:
    errors: list[str] = []
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        errors.append(f"forbidden path component: {relative}")
    if relative.name in FORBIDDEN_BASENAMES:
        errors.append(f"obsolete orchestration file: {relative}")
    if any(relative.name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        errors.append(f"forbidden generated suffix: {relative}")
    return errors


def _check_markdown_links(root: Path, path: Path, text: str) -> list[str]:
    errors: list[str] = []
    for raw in _MARKDOWN_LINK.findall(text):
        target = raw.strip().strip("<>").split("#", 1)[0]
        if not target or "://" in target or target.startswith(("mailto:", "#")):
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            errors.append(f"documentation link escapes repository: {path.relative_to(root)} -> {target}")
            continue
        if not resolved.exists():
            errors.append(f"broken documentation link: {path.relative_to(root)} -> {target}")
    return errors


def check_workspace(
    root: Path,
    manifest_path: Path | None = None,
    *,
    check_release_record: bool = True,
) -> list[str]:
    manifest_path = manifest_path or root / PUBLIC_MANIFEST
    if not manifest_path.is_file():
        return [f"missing public manifest: {manifest_path}"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = tuple(payload.get("files", ()))
    actual = candidate_public_files(root)
    errors: list[str] = []
    if expected != tuple(sorted(set(expected))):
        errors.append("public manifest paths must be unique and sorted")
    missing = sorted(set(expected) - set(actual))
    unknown = sorted(set(actual) - set(expected))
    errors.extend(f"manifest file missing: {path}" for path in missing)
    errors.extend(f"unknown publishable file: {path}" for path in unknown)
    release_record = root / "release-verification.json"
    if check_release_record and release_record.is_file():
        errors.extend(check_release_verification(release_record))
    for relative_text in actual:
        relative = PurePosixPath(relative_text)
        path = root / relative_text
        errors.extend(_check_path(relative))
        if path.stat().st_size > MAX_PUBLIC_BYTES:
            errors.append(f"unapproved file above {MAX_PUBLIC_BYTES} bytes: {relative}")
        secret_types = scan_secret_types(path)
        errors.extend(f"secret pattern {kind}: {relative}" for kind in secret_types)
        data = path.read_bytes()
        if b"\0" in data and path.suffix not in ALLOWED_BINARY_SUFFIXES:
            errors.append(f"unapproved binary file: {relative}")
            continue
        if path.suffix.lower() in {".md", ".toml", ".py", ".json", ".yml", ".yaml", ".txt", ".sh"}:
            text = data.decode("utf-8")
            if _ABSOLUTE_LOCAL.search(text):
                errors.append(f"absolute local path: {relative}")
            if path.suffix.lower() == ".md":
                errors.extend(_check_markdown_links(root, path, text))
    return sorted(set(errors))


def archive_members(path: Path) -> tuple[str, ...]:
    if path.suffix == ".whl" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return tuple(sorted(name for name in archive.namelist() if not name.endswith("/")))
    with tarfile.open(path, "r:*") as archive:
        return tuple(sorted(member.name for member in archive.getmembers() if member.isfile()))


def check_distribution(path: Path, manifest_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    members = archive_members(path)
    manifest_path = manifest_path or PUBLIC_MANIFEST
    public_files: set[str] = set()
    if manifest_path.is_file():
        public_files = set(json.loads(manifest_path.read_text(encoding="utf-8"))["files"])
    is_wheel = path.suffix == ".whl"
    sdist_prefix = ""
    if not is_wheel and members:
        first_parts = {PurePosixPath(name).parts[0] for name in members}
        if len(first_parts) == 1:
            sdist_prefix = next(iter(first_parts)) + "/"
    normalized_members: set[str] = set()
    for name in members:
        relative = PurePosixPath(name)
        errors.extend(_check_path(relative))
        if any(part in {"tests", "evidence", "examples"} for part in relative.parts) and is_wheel:
            errors.append(f"wheel contains non-runtime tree: {name}")
        normalized = name.removeprefix(sdist_prefix)
        normalized_members.add(normalized)
        if is_wheel and normalized.startswith("worldzero/") and public_files and normalized not in public_files:
            errors.append(f"wheel package file absent from public manifest: {normalized}")
        if not is_wheel and public_files:
            generated = normalized in {"PKG-INFO", "setup.cfg"} or normalized.startswith(
                "worldzero_research.egg-info/"
            )
            if not generated and normalized not in public_files:
                errors.append(f"sdist file absent from public manifest: {normalized}")
    if public_files:
        if is_wheel and path.name.startswith("worldzero_research-"):
            expected_runtime = {name for name in public_files if name.startswith("worldzero/")}
            errors.extend(
                f"wheel missing manifested package file: {name}"
                for name in sorted(expected_runtime - normalized_members)
            )
        if not is_wheel:
            errors.extend(
                f"sdist missing public source file: {name}"
                for name in sorted(public_files - normalized_members - DETACHED_RELEASE_FILES)
            )
    return sorted(set(errors))


def write_public_manifest(root: Path, output: Path) -> None:
    files = candidate_public_files(root)
    if output.relative_to(root).as_posix() not in files:
        files = tuple(sorted((*files, output.relative_to(root).as_posix())))
    payload = {"schema": "worldzero-public-files-v1", "files": list(files)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    workspace = sub.add_parser("workspace")
    workspace.add_argument("root", nargs="?", type=Path, default=Path("."))
    workspace.add_argument(
        "--skip-release-record",
        action="store_true",
        help="scan the public tree without recursively validating the detached attestation",
    )
    distribution = sub.add_parser("distribution")
    distribution.add_argument("artifact", type=Path)
    write = sub.add_parser("write-public-manifest")
    write.add_argument("root", nargs="?", type=Path, default=Path("."))
    write.add_argument("--output", type=Path, default=PUBLIC_MANIFEST)
    schemas = sub.add_parser("write-schema-identities")
    schemas.add_argument("root", nargs="?", type=Path, default=Path("."))
    schemas.add_argument(
        "--output", type=Path,
        default=Path("evidence/release/schema-identities.json"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write-schema-identities":
        root = args.root.resolve()
        output = args.output if args.output.is_absolute() else root / args.output
        write_schema_identities(root, output)
        return 0
    if args.command == "write-public-manifest":
        root = args.root.resolve()
        output = args.output if args.output.is_absolute() else root / args.output
        write_public_manifest(root, output)
        return 0
    errors = (
        check_workspace(
            args.root.resolve(),
            check_release_record=not args.skip_release_record,
        )
        if args.command == "workspace"
        else check_distribution(args.artifact)
    )
    if errors:
        for error in errors:
            print(error)
        return 1
    print("hygiene: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
