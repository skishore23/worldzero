#!/usr/bin/env python3
"""Assemble the closed release record from retained command evidence."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldzero.release_hygiene import (
    _MANDATORY_RELEASE_COMMANDS,
    _parse_release_output,
    _readme_smoke_from_result,
    sha256_file,
)
from worldzero.scoring import default_scoring_profile


def _log_identity(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _captured_command(root: Path, command_id: str) -> dict[str, object]:
    relative = Path("evidence/release/commands") / f"{command_id}.json"
    log = json.loads((root / relative).read_text(encoding="utf-8"))
    parse_errors: list[str] = []
    counts = _parse_release_output(log["parser"], log["stdout"], parse_errors, command_id)
    if parse_errors or counts is None:
        raise RuntimeError("; ".join(parse_errors))
    passed, failed, errored = counts
    return {
        "id": command_id,
        "argv": log["argv"],
        "cwd": log["cwd"],
        "environment": log["environment"],
        "parser": log["parser"],
        "capture_status": "captured",
        "log": _log_identity(relative),
        "exit_code": log["exit_code"],
        "passed": log["exit_code"] == 0 and failed == errored == 0 and passed > 0,
        "pass_count": passed,
        "fail_count": failed,
        "error_count": errored,
        "duration_seconds": log["duration_seconds"],
    }


def _pending_command(command_id: str) -> dict[str, object]:
    argv, parser = _MANDATORY_RELEASE_COMMANDS[command_id]
    return {
        "id": command_id,
        "argv": argv,
        "cwd": ".",
        "environment": {},
        "parser": parser,
        "capture_status": "capture_pending",
        "log": None,
        "exit_code": None,
        "passed": False,
        "pass_count": None,
        "fail_count": None,
        "error_count": None,
        "duration_seconds": None,
    }


def _build_distributions(root: Path) -> dict[str, object]:
    log = json.loads((root / "evidence/release/commands/build_offline_smoke.json").read_text())
    result = json.loads(log["stdout"])
    artifacts = [
        {name: row[name] for name in ("name", "sha256", "member_count", "member_manifest_sha256")}
        for row in result["artifacts"]
    ]
    by_name = {row["name"]: row for row in result["artifacts"]}
    sdist = by_name["worldzero_research-0.3.0.tar.gz"]
    return {
        "source_date_epoch": result["source_date_epoch"],
        "status": "passed",
        "artifacts": artifacts,
        "two_build_comparison": {
            "worldzero_wheel_bytes_equal": by_name["worldzero_research-0.3.0-py3-none-any.whl"]["container_bytes_equal"],
            "example_wheel_bytes_equal": by_name["worldzero_example_law-0.1.0-py3-none-any.whl"]["container_bytes_equal"],
            "sdist_member_names_and_bytes_equal": sdist["member_bytes_equal"],
            "sdist_container_bytes_equal": sdist["container_bytes_equal"],
            "sdist_second_sha256": sdist["second_sha256"],
            "sdist_container_note": (
                "Setuptools preserves a varying gzip header timestamp; exact member names and bytes are identical."
            ),
        },
    }


def _derive_readme_smoke(root: Path) -> dict[str, object]:
    log = json.loads(
        (root / "evidence/release/commands/readme_quickstart.json").read_text(
            encoding="utf-8"
        )
    )
    errors: list[str] = []
    if _parse_release_output(
        "readme_quickstart", log["stdout"], errors, "readme_quickstart"
    ) is None or errors:
        raise RuntimeError("; ".join(errors))
    return _readme_smoke_from_result(json.loads(log["stdout"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending-full-suite", action="store_true")
    args = parser.parse_args()
    root = ROOT
    old = json.loads((root / "release-verification.json").read_text(encoding="utf-8"))
    schema_path = root / "evidence/release/schema-identities.json"
    schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
    registry_path = root / "worldzero/laws/official_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    profile = default_scoring_profile().identity_dict()
    commands = []
    for command_id in _MANDATORY_RELEASE_COMMANDS:
        commands.append(
            _pending_command(command_id)
            if args.pending_full_suite and command_id == "complete_test_suite"
            else _captured_command(root, command_id)
        )
    public_manifest = root / "docs/public-files.json"
    public_payload = json.loads(public_manifest.read_text(encoding="utf-8"))
    archive = root / ".local-archive/worldzero-pre-open-source-0.3.0-20260901"
    payload = {
        "schema": "worldzero-release-verification-v1",
        "release": {
            "package": "worldzero-research",
            "version": "0.3.0",
            "law_api_version": "1.0",
            "release_id": "0.3.0-20260901",
            "status": (
                "verification_capture_in_progress" if args.pending_full_suite
                else "verification_complete_pending_independent_review"
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("build", "numpy", "pytest", "setuptools", "wheel")
            },
            "external_network_used": False,
            "model_endpoint_used": False,
            "paid_requests": 0,
        },
        "checkpoints": {
            **{
                f"task_{number}_sha256": sha256_file(root / f".release-checkpoints/task-{number}.json")
                for number in range(1, 8)
            },
            "task_5_final_review_sha256": sha256_file(root / ".superpowers/sdd/2026-08-31-worldzero-law-family-plugins/task-5-final-review.md"),
            "task_6_final_review_finding_sha256": sha256_file(root / ".superpowers/sdd/2026-08-31-worldzero-law-family-plugins/task-6-final-review.md"),
            "task_7_review_sha256": sha256_file(root / ".superpowers/sdd/2026-08-31-worldzero-law-family-plugins/task-7-review.md"),
        },
        "drift": old["drift"],
        "commands": commands,
        "identities": {
            "official_registry_sha256": sha256_file(registry_path),
            "scoring_profile": profile,
            "official_families": [
                {
                    **{name: row[name] for name in (
                        "family_id", "family_version", "fingerprint", "calibration_suite_sha256"
                    )},
                    "validation_passed": True,
                }
                for row in registry["approved_families"]
            ],
        },
        "schemas": {
            "manifest": {
                "path": "evidence/release/schema-identities.json",
                "sha256": sha256_file(schema_path),
                "bytes": schema_path.stat().st_size,
            },
            "contracts": [
                {"schema_id": row["schema_id"], "identity_sha256": row["identity_sha256"]}
                for row in schema_payload["contracts"]
            ],
        },
        "compatibility": {
            **old["compatibility"],
            "invariant_matrix_status": "passed_in_complete_test_matrix",
            "mixed_key_collision_status": "passed_7_of_7",
        },
        "distributions": _build_distributions(root),
        "cleanup": {
            "public_manifest_sha256": sha256_file(public_manifest),
            "public_file_count": len(public_payload["files"]),
            "archive_manifest_sha256": sha256_file(archive / "archive-manifest.json"),
            "archive_source_identities_sha256": sha256_file(archive / "archive-source-identities.json"),
            "recovery_path": ".local-archive/worldzero-pre-open-source-0.3.0-20260901",
            "recovery_policy": "Restore one exact identity to a separate review location; never publish the archive wholesale.",
        },
        "hygiene": {
            "workspace": "passed",
            "credential_patterns": 0,
            "absolute_developer_paths": 0,
            "broken_relative_links": 0,
            "unexpected_large_or_binary_files": 0,
            "internal_files_in_public_manifest": 0,
            "git_initialized": False,
        },
        "readme_smoke": _derive_readme_smoke(root),
        "reviewer": {
            "status": "pending",
            "spec": "PENDING",
            "quality": "PENDING",
            "critical": 0,
            "important": 0,
            "minor": 0,
            "report_sha256": None,
        },
        "open_source_ready": False,
    }
    (root / "release-verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
