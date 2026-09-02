"""Deterministic registry and lazy entry-point loading contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from worldzero.laws import (
    AccountingDelta,
    CalibrationCase,
    ChannelSpec,
    ControlKind,
    ControlSpec,
    ControlSuite,
    DerivedLawState,
    EvaluatorTrace,
    FamilyDescriptor,
    FamilyEvidence,
    FamilyInstance,
    InterventionTransition,
    LawFamily,
    LawTransition,
    ProposalDraw,
    PublicSubstrateView,
    SampleContext,
    SubstrateView,
)
from worldzero.laws.registry import (
    ENTRY_POINT_GROUP,
    DuplicateFamilyError,
    FamilyLoadError,
    FamilyNotFoundError,
    LawRegistry,
    OfficialRegistryError,
    builtin_registry,
    discover_entry_points,
    fingerprint_family,
    resolve_family,
)


def make_descriptor(family_id: str, **overrides: object) -> FamilyDescriptor:
    values: dict[str, object] = {
        "family_id": family_id,
        "api_version": "1.0",
        "family_version": "1.2.3",
        "display_name": family_id,
        "package": "registry-test-plugin",
        "package_version": "2.0.0",
        "capabilities": frozenset(),
        "observation_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    }
    values.update(overrides)
    return FamilyDescriptor(**values)  # type: ignore[arg-type]


class FixtureFamily(LawFamily):
    descriptor = make_descriptor("example_org:fixture")
    fixture_calibration_cases: tuple[CalibrationCase, ...] = ()

    def sample(self, context: SampleContext) -> FamilyInstance:
        return FamilyInstance(self.descriptor.family_id, self.descriptor.family_version, {}, {})

    def channels(self, instance: FamilyInstance, config: Any) -> tuple[ChannelSpec, ...]:
        return ()

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        return DerivedLawState({}, False)

    def apply_proposal(
        self,
        proposal: ProposalDraw,
        view: SubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> LawTransition | None:
        return None

    def project_public(
        self,
        view: PublicSubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> dict[str, object]:
        return {}

    def controls(self, instance: FamilyInstance) -> ControlSuite:
        return ControlSuite(
            null=ControlSpec(ControlKind.NULL),
            knockout=ControlSpec(ControlKind.KNOCKOUT),
            broken=ControlSpec(ControlKind.BROKEN),
            retained=ControlSpec(ControlKind.RETAINED),
        )

    def intervene(
        self, control: ControlKind, view: SubstrateView, instance: FamilyInstance,
    ) -> InterventionTransition:
        return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        return FamilyEvidence({})

    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return self.fixture_calibration_cases


class FakeDistribution:
    def __init__(self, name: str) -> None:
        self.name = name
        self.metadata = {"Name": name}


class FakeEntryPoint:
    def __init__(
        self,
        name: str,
        loaded: object,
        *,
        distribution: str = "fixture-distribution",
        value: str = "fixture.module:family",
    ) -> None:
        self.name = name
        self.value = value
        self.group = ENTRY_POINT_GROUP
        self.dist = FakeDistribution(distribution)
        self._loaded = loaded
        self.load_count = 0

    def load(self) -> object:
        self.load_count += 1
        if isinstance(self._loaded, BaseException):
            raise self._loaded
        return self._loaded


def family(
    family_id: str = "example_org:fixture",
    *,
    calibration_cases: tuple[CalibrationCase, ...] = (),
) -> FixtureFamily:
    value = FixtureFamily()
    value.descriptor = make_descriptor(family_id)  # type: ignore[misc]
    value.fixture_calibration_cases = calibration_cases
    return value


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _manual_calibration_fingerprint(value: LawFamily) -> str:
    payload = [
        {
            "absolute_tolerance": case.absolute_tolerance,
            "case_id": case.case_id,
            "expected": _plain_json(case.expected),
            "kind": case.kind,
            "parameters": _plain_json(case.parameters),
            "relative_tolerance": case.relative_tolerance,
            "samples": case.samples,
        }
        for case in value.calibration_cases()
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def official_record(value: LawFamily, fingerprint: str) -> dict[str, object]:
    descriptor = value.descriptor
    return {
        "family_id": descriptor.family_id,
        "api_version": descriptor.api_version,
        "family_version": descriptor.family_version,
        "package": descriptor.package,
        "package_version": descriptor.package_version,
        "fingerprint": fingerprint,
        "calibration_suite_sha256": _manual_calibration_fingerprint(value),
        "release_status": "approved",
    }


def test_builtin_exact_lookup_sorted_listing_and_duplicate_refusal() -> None:
    zeta = family("example_org:zeta")
    alpha = family("example_org:alpha")
    registry = LawRegistry(builtins=(zeta, alpha))

    assert registry.list_family_ids() == ("example_org:alpha", "example_org:zeta")
    assert registry.resolve("example_org:alpha").family is alpha
    with pytest.raises(FamilyNotFoundError, match="exact family ID"):
        registry.resolve("alpha")
    with pytest.raises(DuplicateFamilyError, match="example_org:alpha"):
        LawRegistry(builtins=(alpha, family("example_org:alpha")))


def test_registry_rejects_incompatible_api_before_resolution() -> None:
    incompatible = family()
    object.__setattr__(incompatible.descriptor, "api_version", "2.0")
    with pytest.raises(ValueError, match="API"):
        LawRegistry(builtins=(incompatible,))


def test_catalogue_is_lazy_and_exact_selection_loads_only_selected_entry_point() -> None:
    alpha = FakeEntryPoint("example_org:alpha", lambda: family("example_org:alpha"))
    beta = FakeEntryPoint("example_org:beta", lambda: family("example_org:beta"))
    registry = LawRegistry(entry_points=(beta, alpha))

    assert registry.list_family_ids() == ("example_org:alpha", "example_org:beta")
    assert (alpha.load_count, beta.load_count) == (0, 0)

    selected = registry.resolve("example_org:beta")
    assert selected.family.descriptor.family_id == "example_org:beta"
    assert selected.official is False
    assert selected.origin == "entry-point:fixture-distribution:example_org:beta"
    assert (alpha.load_count, beta.load_count) == (0, 1)
    assert registry.resolve("example_org:beta") is selected
    assert beta.load_count == 1


def test_selected_import_failure_is_isolated_and_names_distribution_and_entry_point() -> None:
    broken = FakeEntryPoint(
        "example_org:broken", ImportError("missing optional dependency"), distribution="broken-dist"
    )
    unrelated = FakeEntryPoint("example_org:unrelated", lambda: family("example_org:unrelated"))
    registry = LawRegistry(entry_points=(unrelated, broken))

    with pytest.raises(FamilyLoadError) as caught:
        registry.resolve("example_org:broken")

    message = str(caught.value)
    assert "broken-dist" in message
    assert "example_org:broken" in message
    assert broken.load_count == 1
    assert unrelated.load_count == 0


def test_entry_point_accepts_instance_or_zero_argument_factory_only() -> None:
    instance = family("example_org:instance")
    assert LawRegistry(
        entry_points=(FakeEntryPoint("example_org:instance", instance),)
    ).resolve("example_org:instance").family is instance

    with pytest.raises(FamilyLoadError, match="LawFamily"):
        LawRegistry(
            entry_points=(FakeEntryPoint("example_org:invalid", lambda: object()),)
        ).resolve("example_org:invalid")


def test_entry_point_descriptor_must_match_exact_entry_point_name() -> None:
    entry = FakeEntryPoint("example_org:advertised", lambda: family("example_org:different"))
    with pytest.raises(FamilyLoadError, match="descriptor ID"):
        LawRegistry(entry_points=(entry,)).resolve("example_org:advertised")


def test_duplicate_entry_point_or_builtin_name_fails_closed_without_importing() -> None:
    first = FakeEntryPoint("example_org:duplicate", lambda: family("example_org:duplicate"))
    second = FakeEntryPoint("example_org:duplicate", lambda: family("example_org:duplicate"))
    with pytest.raises(DuplicateFamilyError, match="example_org:duplicate"):
        LawRegistry(entry_points=(first, second))
    assert first.load_count == second.load_count == 0

    entry = FakeEntryPoint("example_org:fixture", lambda: family())
    with pytest.raises(DuplicateFamilyError, match="example_org:fixture"):
        LawRegistry(builtins=(FixtureFamily(),), entry_points=(entry,))
    assert entry.load_count == 0


def test_fingerprint_is_descriptor_canonical_json_plus_implementation_module_bytes() -> None:
    value = FixtureFamily()
    descriptor_json = json.dumps(
        value.descriptor.persistence_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    module_bytes = Path(__file__).read_bytes()
    expected = hashlib.sha256(descriptor_json + b"\0" + module_bytes).hexdigest()

    assert fingerprint_family(value) == expected
    assert fingerprint_family(value) == fingerprint_family(value)
    changed = family("example_org:changed")
    assert fingerprint_family(changed) != expected


def _calibration_suite() -> tuple[CalibrationCase, ...]:
    return (
        CalibrationCase(
            "rate",
            "analytic",
            {"estimate": 1.0},
            absolute_tolerance=0.1,
            relative_tolerance=0.2,
            samples=10,
            parameters={"pair": [0, 1]},
        ),
        CalibrationCase(
            "accounting",
            "invariant",
            True,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            samples=3,
            parameters={"mode": "strict"},
        ),
    )


def test_calibration_suite_fingerprint_is_public_deterministic_and_canonical() -> None:
    import worldzero.laws as laws
    import worldzero.laws.registry as registry_module

    first = family(calibration_cases=_calibration_suite())
    second = family(calibration_cases=_calibration_suite())
    expected = _manual_calibration_fingerprint(first)

    assert registry_module.calibration_suite_fingerprint(first) == expected
    assert registry_module.calibration_suite_fingerprint(second) == expected
    assert laws.calibration_suite_fingerprint is registry_module.calibration_suite_fingerprint


@pytest.mark.parametrize(
    "changed_suite",
    [
        (
            CalibrationCase(
                "rate", "analytic", {"estimate": 1.1}, 0.1, 0.2, 10, {"pair": [0, 1]}
            ),
            _calibration_suite()[1],
        ),
        tuple(reversed(_calibration_suite())),
        (
            CalibrationCase(
                "rate", "analytic", {"estimate": 1.0}, 0.1, 0.2, 10, {"pair": [1, 0]}
            ),
            _calibration_suite()[1],
        ),
        (
            CalibrationCase(
                "rate", "analytic", {"estimate": 1.0}, 0.11, 0.2, 10, {"pair": [0, 1]}
            ),
            _calibration_suite()[1],
        ),
        (
            CalibrationCase(
                "rate", "analytic", {"estimate": 1.0}, 0.1, 0.21, 10, {"pair": [0, 1]}
            ),
            _calibration_suite()[1],
        ),
        (
            CalibrationCase(
                "rate", "analytic", {"estimate": 1.0}, 0.1, 0.2, 11, {"pair": [0, 1]}
            ),
            _calibration_suite()[1],
        ),
    ],
    ids=("case", "order", "parameter", "absolute-tolerance", "relative-tolerance", "samples"),
)
def test_calibration_suite_fingerprint_covers_every_identity_dimension(
    changed_suite: tuple[CalibrationCase, ...],
) -> None:
    import worldzero.laws.registry as registry_module

    baseline = registry_module.calibration_suite_fingerprint(
        family(calibration_cases=_calibration_suite())
    )
    assert registry_module.calibration_suite_fingerprint(
        family(calibration_cases=changed_suite)
    ) != baseline


def test_calibration_suite_fingerprint_rejects_malformed_or_nonfinite_cases() -> None:
    import worldzero.laws.registry as registry_module

    wrong_container = family()
    wrong_container.fixture_calibration_cases = []  # type: ignore[assignment]
    with pytest.raises(TypeError, match="tuple"):
        registry_module.calibration_suite_fingerprint(wrong_container)

    wrong_item = family()
    wrong_item.fixture_calibration_cases = (object(),)  # type: ignore[assignment]
    with pytest.raises(TypeError, match="CalibrationCase"):
        registry_module.calibration_suite_fingerprint(wrong_item)

    nonfinite = CalibrationCase("rate", "analytic", 1.0)
    object.__setattr__(nonfinite, "absolute_tolerance", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        registry_module.calibration_suite_fingerprint(
            family(calibration_cases=(nonfinite,))
        )


def test_unlisted_family_is_experimental_and_exact_official_match_is_official() -> None:
    value = family(calibration_cases=_calibration_suite())
    fingerprint = fingerprint_family(value)

    experimental = LawRegistry(builtins=(value,), official_records=()).resolve(value.descriptor.family_id)
    assert experimental.official is False

    official = LawRegistry(
        builtins=(value,), official_records=(official_record(value, fingerprint),)
    ).resolve(value.descriptor.family_id)
    assert official.official is True
    assert official.fingerprint == fingerprint


def test_official_allowlist_refuses_calibration_suite_drift() -> None:
    value = family(calibration_cases=_calibration_suite())
    record = official_record(value, fingerprint_family(value))
    record["calibration_suite_sha256"] = "0" * 64

    with pytest.raises(OfficialRegistryError, match="calibration.*drift"):
        LawRegistry(builtins=(value,), official_records=(record,)).resolve(
            value.descriptor.family_id
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("api_version", "0.9"),
        ("family_version", "9.9.9"),
        ("package_version", "9.9.9"),
        ("fingerprint", "0" * 64),
    ],
)
def test_official_allowlist_refuses_api_version_package_or_fingerprint_drift(
    field: str, replacement: str,
) -> None:
    value = FixtureFamily()
    record = official_record(value, fingerprint_family(value))
    record[field] = replacement
    with pytest.raises(OfficialRegistryError, match="drift"):
        LawRegistry(builtins=(value,), official_records=(record,)).resolve(value.descriptor.family_id)


def test_official_allowlist_rejects_duplicate_rows() -> None:
    value = FixtureFamily()
    record = official_record(value, fingerprint_family(value))
    with pytest.raises(OfficialRegistryError, match="duplicate"):
        LawRegistry(builtins=(value,), official_records=(record, dict(record)))


def test_discovery_filters_group_sorts_without_loading_and_records_exact_group(monkeypatch: pytest.MonkeyPatch) -> None:
    alpha = FakeEntryPoint("example_org:alpha", lambda: family("example_org:alpha"))
    beta = FakeEntryPoint("example_org:beta", lambda: family("example_org:beta"))
    calls: list[str] = []

    def fake_entry_points(*, group: str) -> tuple[FakeEntryPoint, ...]:
        calls.append(group)
        return (beta, alpha)

    monkeypatch.setattr("worldzero.laws.registry.metadata.entry_points", fake_entry_points)
    assert discover_entry_points() == (alpha, beta)
    assert calls == [ENTRY_POINT_GROUP]
    assert alpha.load_count == beta.load_count == 0


def test_builtin_registry_does_not_query_package_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("package metadata lookup is forbidden for built-ins")

    monkeypatch.setattr("worldzero.laws.registry.metadata.entry_points", forbidden)
    registry = builtin_registry()
    assert registry.list_family_ids() == (
        "worldzero:catalysis",
        "worldzero:delayed-transformation",
        "worldzero:inhibition",
        "worldzero:null",
    )
    assert registry.resolve("worldzero:catalysis").origin == "builtin"
    assert registry.resolve("worldzero:null").origin == "builtin"


def test_public_resolve_helper_uses_given_registry() -> None:
    value = FixtureFamily()
    registry = LawRegistry(builtins=(value,))
    assert resolve_family("example_org:fixture", registry=registry) is registry.resolve(
        "example_org:fixture"
    )


def test_registry_helpers_are_exported_from_public_laws_package() -> None:
    import worldzero.laws as laws

    assert laws.LawRegistry is LawRegistry
    assert laws.builtin_registry is builtin_registry
    assert laws.discover_entry_points is discover_entry_points
    assert laws.resolve_family is resolve_family


def test_package_metadata_declares_v030_and_bundles_official_registry() -> None:
    import worldzero

    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    assert worldzero.__version__ == "0.3.0"
    assert project["project"]["version"] == "0.3.0"
    assert project["project"]["dependencies"] == ["numpy>=1.26,<3"]
    assert "tomli>=2; python_version < '3.11'" in project["project"]["optional-dependencies"]["test"]
    assert "Programming Language :: Python :: 3" in project["project"]["classifiers"]
    assert "laws/official_registry.json" in project["tool"]["setuptools"]["package-data"]["worldzero"]
