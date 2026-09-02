#!/usr/bin/env python3
"""Build twice from the public allowlist and exercise the wheels offline."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


EPOCH = "1704067200"
DETACHED_PREFIXES = ("evidence/release/commands/",)
DETACHED_FILES = {"release-verification.json"}


def _run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None,
         expected: int = 0) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    result = subprocess.run(
        argv, cwd=cwd, env=child_env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != expected:
        raise RuntimeError(
            f"command failed ({result.returncode}, expected {expected}): {argv!r}\n"
            f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
        )
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member_manifest(path: Path) -> tuple[int, str]:
    rows: list[dict[str, str]] = []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(n for n in archive.namelist() if not n.endswith("/")):
                rows.append({"name": name, "sha256": hashlib.sha256(archive.read(name)).hexdigest()})
    else:
        with tarfile.open(path, "r:*") as archive:
            members = sorted((m for m in archive.getmembers() if m.isfile()), key=lambda m: m.name)
            for member in members:
                handle = archive.extractfile(member)
                assert handle is not None
                rows.append({"name": member.name, "sha256": hashlib.sha256(handle.read()).hexdigest()})
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(rows), hashlib.sha256(encoded).hexdigest()


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
    # The attestation and its command logs are detached from the archives they
    # authenticate.  Give each clean build source its own exact allowlist.
    (destination / "docs/public-files.json").write_text(
        json.dumps({"schema": "worldzero-public-files-v1", "files": copied}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="worldzero-release-a-") as a_text, \
         tempfile.TemporaryDirectory(prefix="worldzero-release-b-") as b_text:
        builds: list[Path] = []
        for text in (a_text, b_text):
            work = Path(text)
            source = work / "source"
            dist = work / "dist"
            _copy_public(root, source)
            _run([sys.executable, "-m", "worldzero.release_hygiene", "workspace", "."], cwd=source)
            _run(
                [sys.executable, "-m", "build", "--no-isolation", "--outdir", "../dist"],
                cwd=source, env={"SOURCE_DATE_EPOCH": EPOCH},
            )
            _run(
                [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", "../../../dist"],
                cwd=source / "examples/community_law_plugin",
                env={"SOURCE_DATE_EPOCH": EPOCH},
            )
            builds.append(work)

        artifact_names = (
            "worldzero_research-0.3.0-py3-none-any.whl",
            "worldzero_example_law-0.1.0-py3-none-any.whl",
            "worldzero_research-0.3.0.tar.gz",
        )
        artifacts: list[dict[str, object]] = []
        for name in artifact_names:
            first = builds[0] / "dist" / name
            second = builds[1] / "dist" / name
            for artifact in (first, second):
                _run([sys.executable, "-m", "worldzero.release_hygiene", "distribution", str(artifact)], cwd=root)
            first_count, first_manifest = _member_manifest(first)
            second_count, second_manifest = _member_manifest(second)
            if (first_count, first_manifest) != (second_count, second_manifest):
                raise RuntimeError(f"member manifest drift: {name}")
            artifacts.append({
                "name": name,
                "sha256": _sha(first),
                "second_sha256": _sha(second),
                "member_count": first_count,
                "member_manifest_sha256": first_manifest,
                "container_bytes_equal": _sha(first) == _sha(second),
                "member_bytes_equal": True,
            })

        venv = builds[0] / "venv"
        _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], cwd=root)
        python = venv / "bin/python"
        worldzero = venv / "bin/worldzero"
        wheels = [builds[0] / "dist" / artifact_names[0], builds[0] / "dist" / artifact_names[1]]
        _run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", *(str(p) for p in wheels)], cwd=root)
        listed = _run([str(worldzero), "laws", "list"], cwd=root)
        listing = json.loads(listed.stdout)
        expected_ids = [
            "example_org:preserver", "worldzero:catalysis",
            "worldzero:delayed-transformation", "worldzero:inhibition", "worldzero:null",
        ]
        if listing["family_ids"] != expected_ids:
            raise RuntimeError("installed family listing mismatch")
        validated = _run([
            str(worldzero), "laws", "validate", "example_org:preserver",
            "--experimental-family", "--seeds", "1",
        ], cwd=root)
        if not json.loads(validated.stdout)["passed"]:
            raise RuntimeError("example validation failed")
        _run([str(worldzero), "laws", "inspect", "example_org:preserver"], cwd=root, expected=2)
        demo = builds[0] / "experimental-demo"
        _run([
            str(worldzero), "demo", "--seeds", "1", "--law-family",
            "example_org:preserver", "--experimental-family", "--output", str(demo),
        ], cwd=root)
        replayed = _run([
            str(worldzero), "replay",
            str(demo / "traces/pressure-experimenter/1452232541.json.gz"),
        ], cwd=root)
        replay = json.loads(replayed.stdout)
        if not replay["verified"]:
            raise RuntimeError("experimental replay failed")
        print(json.dumps({
            "schema": "worldzero-build-offline-smoke-v1",
            "checks": [
                "clean_public_source_a",
                "clean_public_source_b",
                "worldzero_wheel_hygiene_a",
                "worldzero_wheel_hygiene_b",
                "example_wheel_hygiene_a",
                "example_wheel_hygiene_b",
                "source_distribution_hygiene_a",
                "source_distribution_hygiene_b",
                "two_build_member_identity",
                "offline_wheel_install",
                "installed_family_listing",
                "community_example_validation",
                "official_identity_refusal",
                "experimental_episode_replay",
            ],
            "source_date_epoch": int(EPOCH),
            "artifacts": artifacts,
            "offline_install": True,
            "family_ids": expected_ids,
            "example_validation": True,
            "official_refusal": True,
            "experimental_replay": replay,
            "external_network_used": False,
            "model_endpoint_used": False,
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
