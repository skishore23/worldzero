"""Command-line contracts for exact law-family selection."""

from __future__ import annotations

import json
import sys

import pytest

from worldzero import cli
from worldzero.laws import CalibrationCase, FamilyDescriptor, FamilyInstance, SampleContext
from worldzero.laws.builtin.null import NullFamily
from worldzero.laws.registry import ENTRY_POINT_GROUP


class MinimalFamily(NullFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:minimal",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Minimal CLI fixture",
        package="worldzero-cli-fixture",
        package_version="1.0.0",
        capabilities=frozenset({"geometry_control"}),
        observation_schema={
            "type": "object", "additionalProperties": False, "properties": {},
        },
    )

    def sample(self, context: SampleContext) -> FamilyInstance:
        base = super().sample(context)
        return FamilyInstance(
            self.descriptor.family_id,
            self.descriptor.family_version,
            base.hidden_parameters,
            {},
        )

    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return (CalibrationCase("sdk-smoke", "invariant", True),)


class EmptyCalibrationFamily(MinimalFamily):
    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return ()


def invoke(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str):
    monkeypatch.setattr(sys, "argv", ["worldzero", *args])
    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_laws_list_is_closed_sorted_json(monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, stderr = invoke(monkeypatch, capsys, "laws", "list")

    assert code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert set(payload) == {"schema", "family_ids"}
    assert payload["schema"] == "worldzero-law-list-v1"
    assert payload["family_ids"] == sorted(payload["family_ids"])
    assert payload["family_ids"] == [
        "worldzero:catalysis",
        "worldzero:delayed-transformation",
        "worldzero:inhibition",
        "worldzero:null",
    ]


def test_laws_inspect_returns_exact_identity(monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, stderr = invoke(
        monkeypatch, capsys, "laws", "inspect", "worldzero:null"
    )

    assert code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert set(payload) == {
        "schema", "family_id", "descriptor", "fingerprint",
        "calibration_suite_sha256", "origin", "official", "experimental",
    }
    assert payload["schema"] == "worldzero-law-inspection-v1"
    assert payload["family_id"] == "worldzero:null"
    assert payload["official"] is True
    assert payload["experimental"] is False


def test_laws_validate_exit_status_tracks_machine_report(monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, stderr = invoke(
        monkeypatch, capsys, "laws", "validate", "worldzero:null", "--seeds", "1"
    )

    assert code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["schema"] == "worldzero-family-validation-v1"
    assert payload["passed"] is True
    assert payload["seed_count"] == 1


def test_run_rejects_non_namespaced_law_family_before_manifest_access(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path,
) -> None:
    code, stdout, stderr = invoke(
        monkeypatch,
        capsys,
        "run",
        "--manifest",
        str(tmp_path / "missing.json"),
        "--name",
        "x",
        "--law-family",
        "catalysis",
    )

    assert code == 2
    assert stdout == ""
    assert "exact namespaced" in stderr


def test_experimental_flag_does_not_relax_paid_model_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path,
) -> None:
    manifest = tmp_path / "protocol.json"
    from worldzero.protocol import create_manifest

    create_manifest(manifest, seed=5, dev=1, test=1)
    code, stdout, stderr = invoke(
        monkeypatch,
        capsys,
        "run",
        "--manifest",
        str(manifest),
        "--name",
        "x",
        "--policy",
        "llm",
        "--law-family",
        "worldzero:null",
        "--experimental-family",
    )

    assert code == 2
    assert stdout == ""
    assert "--model is required" in stderr


class FakeDistribution:
    name = "worldzero-example-fixture"
    metadata = {"Name": name}


class FakeEntryPoint:
    group = ENTRY_POINT_GROUP
    dist = FakeDistribution()

    def __init__(self, name: str, loaded: object) -> None:
        self.name = name
        self.value = "fixture:family"
        self.loaded = loaded
        self.load_count = 0

    def load(self) -> object:
        self.load_count += 1
        if isinstance(self.loaded, BaseException):
            raise self.loaded
        return self.loaded


def _entry_points(monkeypatch: pytest.MonkeyPatch, *points: FakeEntryPoint) -> None:
    monkeypatch.setattr(
        "worldzero.laws.registry.metadata.entry_points",
        lambda **kwargs: tuple(points),
    )


def test_laws_list_never_imports_installed_plugins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    broken = FakeEntryPoint("example_org:broken", ImportError("do not import"))
    _entry_points(monkeypatch, broken)

    code, stdout, stderr = invoke(monkeypatch, capsys, "laws", "list")

    assert code == 0
    assert stderr == ""
    assert "example_org:broken" in json.loads(stdout)["family_ids"]
    assert broken.load_count == 0


def test_exact_community_selection_warns_before_loading_and_loads_no_unrelated_plugin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    selected = FakeEntryPoint("example_org:minimal", MinimalFamily())
    unrelated = FakeEntryPoint("example_org:unrelated", ImportError("must stay lazy"))
    warning_seen_at_load: list[bool] = []
    original_load = selected.load

    def load_selected() -> object:
        warning_seen_at_load.append("trusted in-process Python" in capsys.readouterr().err)
        return original_load()

    monkeypatch.setattr(selected, "load", load_selected)
    _entry_points(monkeypatch, unrelated, selected)

    code, stdout, stderr = invoke(
        monkeypatch,
        capsys,
        "laws",
        "inspect",
        "example_org:minimal",
        "--experimental-family",
    )

    assert code == 0
    assert stderr == ""
    assert warning_seen_at_load == [True]
    assert json.loads(stdout)["experimental"] is True
    assert selected.load_count == 1
    assert unrelated.load_count == 0


def test_community_selection_without_flag_refuses_before_import(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    selected = FakeEntryPoint("example_org:minimal", MinimalFamily())
    _entry_points(monkeypatch, selected)

    code, stdout, stderr = invoke(
        monkeypatch, capsys, "laws", "inspect", "example_org:minimal"
    )

    assert code == 2
    assert stdout == ""
    assert "--experimental-family" in stderr
    assert selected.load_count == 0


def test_selected_plugin_failure_isolated_without_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    selected = FakeEntryPoint("example_org:broken", ImportError("selected exploded"))
    unrelated = FakeEntryPoint("example_org:unrelated", MinimalFamily())
    _entry_points(monkeypatch, unrelated, selected)

    code, stdout, stderr = invoke(
        monkeypatch,
        capsys,
        "laws",
        "inspect",
        "example_org:broken",
        "--experimental-family",
    )

    assert code == 2
    assert stdout == ""
    assert "selected exploded" in stderr
    assert selected.load_count == 1
    assert unrelated.load_count == 0


def test_validation_failure_is_json_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    selected = FakeEntryPoint("example_org:minimal", EmptyCalibrationFamily())
    _entry_points(monkeypatch, selected)

    code, stdout, stderr = invoke(
        monkeypatch,
        capsys,
        "laws",
        "validate",
        "example_org:minimal",
        "--experimental-family",
        "--seeds",
        "1",
    )

    assert code == 1
    assert json.loads(stdout)["passed"] is False
    assert "trusted in-process Python" in stderr


def test_experimental_run_freezes_identity_and_official_mode_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    from worldzero.protocol import create_manifest, execute, read_trace
    from worldzero.laws.registry import installed_registry

    selected = FakeEntryPoint("example_org:minimal", MinimalFamily())
    _entry_points(monkeypatch, selected)
    registry = installed_registry()
    manifest = create_manifest(tmp_path / "protocol.json", seed=8, dev=1, test=1)
    manifest["conditions"]["pressure"]["max_decisions"] = 1

    with pytest.raises(ValueError, match="experimental_family"):
        execute(
            manifest,
            output=tmp_path / "refused",
            name="refused",
            include_inheritance=False,
            law_family="example_org:minimal",
            family_registry=registry,
            progress=False,
        )

    summary = execute(
        manifest,
        output=tmp_path / "accepted",
        name="accepted",
        include_inheritance=False,
        capture_first=1,
        law_family="example_org:minimal",
        experimental_family=True,
        family_registry=registry,
        progress=False,
    )
    identity = summary["specification"]["family_identity"]
    assert summary["specification"]["experimental_family"] is True
    assert identity["experimental"] is True
    assert identity["official"] is False
    assert identity["origin"].startswith("entry-point:")
    trace = read_trace(
        tmp_path / "accepted" / "traces" / "accepted" /
        f"{manifest['dev_seeds'][0]}.json.gz"
    )
    assert trace["family_identity"]["experimental"] is True
    assert trace["family_identity"]["official"] is False
