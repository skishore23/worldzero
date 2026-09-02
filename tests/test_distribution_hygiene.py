from __future__ import annotations

import copy
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

from worldzero.release_hygiene import (
    archive_members,
    check_distribution,
    check_release_verification,
    check_workspace,
    _parse_release_output,
    sha256_file,
    tree_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_workspace_matches_exact_manifest_and_hygiene_contract() -> None:
    assert check_workspace(ROOT) == []


def test_required_public_files_are_explicitly_tracked() -> None:
    files = set(json.loads((ROOT / "docs/public-files.json").read_text())["files"])
    required = {
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "docs/ARCHITECTURE.md",
        "docs/HISTORY.md",
        "docs/SUPPORT.md",
        "docs/THIRD_PARTY.md",
        "docs/CONTRIBUTING_LAWS.md",
        "docs/RELEASE_CHECKLIST.md",
        "evidence/reference/README.md",
        "evidence/reference/manifest.json",
        "evidence/release/schema-identities.json",
        "evidence/release/commands/complete_test_suite.json",
        "evidence/release/commands/readme_quickstart.json",
        "examples/community_law_plugin/pyproject.toml",
        "scripts/release/capture_command.py",
        "scripts/release/verify_readme_quickstart.py",
        "scripts/release/build_offline_smoke.py",
        "scripts/release/assemble_record.py",
        "worldzero/laws/testing.py",
        "worldzero/laws/registry.py",
        "worldzero/laws/official_registry.json",
        "release-verification.json",
    }
    assert required <= files


def test_tree_identity_is_path_ordered_and_sensitive(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "b").write_bytes(b"two")
    (tree / "a").write_bytes(b"one")
    first = tree_identity(tree)
    assert first["files"] == 2
    assert first["bytes"] == 6
    (tree / "a").write_bytes(b"ONE")
    assert tree_identity(tree)["sha256"] != first["sha256"]
    assert sha256_file(tree / "b") == "3fc4ccfe745870e2c0d99f71f30ff0656c8dedd41cc1d7d3d376b0dbe685e2f3"


def test_distribution_checker_rejects_internal_and_generated_members(tmp_path: Path) -> None:
    wheel = tmp_path / "bad.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("worldzero/__init__.py", "")
        archive.writestr(".superpowers/private.md", "")
        archive.writestr("worldzero/__pycache__/bad.pyc", b"\0")
    manifest = tmp_path / "public.json"
    manifest.write_text(json.dumps({"files": ["worldzero/__init__.py"]}))
    errors = check_distribution(wheel, manifest)
    assert any(".superpowers" in error for error in errors)
    assert any("__pycache__" in error for error in errors)


def test_sdist_member_listing_is_deterministic(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in (("pkg/z.txt", b"z"), ("pkg/a.txt", b"a")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    assert archive_members(archive_path) == ("pkg/a.txt", "pkg/z.txt")


def test_distribution_checker_requires_manifested_members(tmp_path: Path) -> None:
    wheel = tmp_path / "worldzero_research-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("worldzero/__init__.py", "")
        archive.writestr("worldzero/unlisted.py", "")
    manifest = tmp_path / "public.json"
    manifest.write_text(json.dumps({"files": ["worldzero/__init__.py", "worldzero/missing.py"]}))
    errors = check_distribution(wheel, manifest)
    assert "wheel package file absent from public manifest: worldzero/unlisted.py" in errors
    assert "wheel missing manifested package file: worldzero/missing.py" in errors


def test_detached_release_record_can_authenticate_sdist_without_self_membership(
    tmp_path: Path,
) -> None:
    sdist = tmp_path / "worldzero_research-0.3.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        data = b"source"
        info = tarfile.TarInfo("worldzero_research-0.3.0/worldzero/__init__.py")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    manifest = tmp_path / "public.json"
    manifest.write_text(json.dumps({
        "files": ["release-verification.json", "worldzero/__init__.py"]
    }))
    assert check_distribution(sdist, manifest) == []


def test_release_verification_schema_is_closed_and_finite() -> None:
    record = ROOT / "release-verification.json"
    assert check_release_verification(record) == []
    payload = json.loads(record.read_text())
    payload["unexpected"] = True
    changed = record.parent / ".release-verification-invalid.json"
    try:
        changed.write_text(json.dumps(payload))
        assert any("not closed" in error for error in check_release_verification(changed))
    finally:
        changed.unlink(missing_ok=True)


def test_release_verification_accepts_clean_public_checkout_without_private_roots(
    tmp_path: Path,
) -> None:
    public = tmp_path / "worldzero"
    manifest = json.loads((ROOT / "docs/public-files.json").read_text())
    for relative in manifest["files"]:
        source = ROOT / relative
        destination = public / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    assert not (public / ".local-archive").exists()
    assert not (public / ".release-checkpoints").exists()
    assert not (public / ".superpowers").exists()
    assert check_release_verification(public / "release-verification.json") == []


def _release_mutations(payload: dict[str, object]):
    mutations = []

    changed = copy.deepcopy(payload)
    changed["environment"]["unexpected"] = True
    mutations.append(changed)

    changed = copy.deepcopy(payload)
    del changed["identities"]["official_registry_sha256"]
    mutations.append(changed)

    changed = copy.deepcopy(payload)
    changed["distributions"]["artifacts"][0]["sha256"] = "not-a-digest"
    mutations.append(changed)

    changed = copy.deepcopy(payload)
    changed["commands"][0]["pass_count"] = 999_999
    mutations.append(changed)

    changed = copy.deepcopy(payload)
    changed["checkpoints"]["task_1_sha256"] = "x"
    mutations.append(changed)

    changed = copy.deepcopy(payload)
    changed["release"]["unexpected"] = "field"
    mutations.append(changed)

    changed = copy.deepcopy(payload)
    del changed["distributions"]["artifacts"][0]["member_manifest_sha256"]
    mutations.append(changed)

    changed = copy.deepcopy(payload)
    changed["commands"][0]["passed"] = not changed["commands"][0]["passed"]
    mutations.append(changed)

    return mutations


def test_release_verification_rejects_reviewed_and_nested_false_certificates(
    tmp_path: Path,
) -> None:
    payload = json.loads((ROOT / "release-verification.json").read_text())
    mutations = _release_mutations(payload)
    assert len(mutations) == 8
    for index, changed_payload in enumerate(mutations):
        path = tmp_path / f"changed-{index}.json"
        path.write_text(json.dumps(changed_payload))
        assert check_release_verification(path), f"mutation {index} false-certified"


def test_readme_smoke_is_derived_from_retained_named_checks() -> None:
    assembler = (ROOT / "scripts/release/assemble_record.py").read_text()
    assert "def _derive_readme_smoke" in assembler
    for field in (
        "source_install",
        "four_builtin_workflow",
        "demo_replay",
        "example_build_install_validate",
        "experimental_episode_replay",
        "official_refusal",
    ):
        assert f'"{field}": True' not in assembler


def test_readme_quickstart_parser_rejects_missing_extra_and_false_checks() -> None:
    log = json.loads(
        (ROOT / "evidence/release/commands/readme_quickstart.json").read_text()
    )
    result = json.loads(log["stdout"])
    mutations = []
    changed = copy.deepcopy(result)
    changed["checks"].pop()
    mutations.append(changed)
    changed = copy.deepcopy(result)
    changed["checks"][0]["unexpected"] = True
    mutations.append(changed)
    changed = copy.deepcopy(result)
    changed["checks"][0]["passed"] = False
    mutations.append(changed)
    for index, changed in enumerate(mutations):
        errors: list[str] = []
        parsed = _parse_release_output(
            "readme_quickstart", json.dumps(changed), errors, f"mutation[{index}]"
        )
        assert parsed is None
        assert errors
