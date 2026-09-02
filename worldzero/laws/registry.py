"""Deterministic built-in and installed law-family registry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

from .base import LawFamily
from .types import (
    CalibrationCase,
    FamilyDescriptor,
    SUPPORTED_API_VERSION,
    _require_family_id,
    thaw_json,
)


ENTRY_POINT_GROUP = "worldzero.law_families"


class LawRegistryError(ValueError):
    """Base class for deterministic registry failures."""


class DuplicateFamilyError(LawRegistryError):
    """Two registrations advertise the same exact family ID."""


class FamilyNotFoundError(LawRegistryError):
    """No registration advertises the requested exact family ID."""


class FamilyLoadError(LawRegistryError):
    """A selected entry point could not produce a conforming family."""


class OfficialRegistryError(LawRegistryError):
    """An official allowlist row is invalid or no longer matches source."""


@dataclass(frozen=True)
class RegisteredFamily:
    family: LawFamily
    origin: str
    official: bool
    fingerprint: str

    @property
    def experimental(self) -> bool:
        """Whether this exact implementation is outside the official allowlist."""

        return not self.official


def _canonical_descriptor_bytes(descriptor: FamilyDescriptor) -> bytes:
    return json.dumps(
        descriptor.persistence_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _implementation_module_bytes(family: LawFamily) -> bytes:
    module_name = family.__class__.__module__
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None) if module is not None else None
    if not module_file:
        raise FamilyLoadError(
            f"Cannot fingerprint {family.descriptor.family_id}: implementation module "
            f"{module_name!r} has no readable file"
        )
    try:
        return Path(module_file).read_bytes()
    except OSError as exc:
        raise FamilyLoadError(
            f"Cannot fingerprint {family.descriptor.family_id}: implementation module "
            f"{module_name!r} cannot be read"
        ) from exc


def fingerprint_family(family: LawFamily) -> str:
    """Hash canonical descriptor JSON and implementation module bytes."""

    _validate_family(family)
    payload = _canonical_descriptor_bytes(family.descriptor)
    return hashlib.sha256(payload + b"\0" + _implementation_module_bytes(family)).hexdigest()


def calibration_suite_fingerprint(family: LawFamily) -> str:
    """Hash every calibration-case field in declared tuple order."""

    value = _validate_family(family)
    cases = value.calibration_cases()
    if type(cases) is not tuple:
        raise TypeError("LawFamily calibration_cases() must return a tuple")
    payload: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, CalibrationCase):
            raise TypeError(
                f"LawFamily calibration_cases()[{index}] must be a CalibrationCase"
            )
        validated = CalibrationCase(
            case_id=case.case_id,
            kind=case.kind,
            expected=case.expected,
            absolute_tolerance=case.absolute_tolerance,
            relative_tolerance=case.relative_tolerance,
            samples=case.samples,
            parameters=case.parameters,
        )
        payload.append(
            {
                "absolute_tolerance": validated.absolute_tolerance,
                "case_id": validated.case_id,
                "expected": thaw_json(validated.expected),
                "kind": validated.kind,
                "parameters": thaw_json(validated.parameters),
                "relative_tolerance": validated.relative_tolerance,
                "samples": validated.samples,
            }
        )
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_family(family: object) -> LawFamily:
    if not isinstance(family, LawFamily):
        raise TypeError("registration must provide a LawFamily instance")
    if not isinstance(family.descriptor, FamilyDescriptor):
        raise TypeError("LawFamily descriptor must be a FamilyDescriptor")
    if family.descriptor.api_version != SUPPORTED_API_VERSION:
        raise ValueError(
            f"LawFamily API {family.descriptor.api_version!r} is incompatible with "
            f"supported API {SUPPORTED_API_VERSION!r}"
        )
    return family


def _distribution_name(entry_point: object) -> str:
    distribution = getattr(entry_point, "dist", None)
    name = getattr(distribution, "name", None)
    if isinstance(name, str) and name:
        return name
    metadata_value = getattr(distribution, "metadata", None)
    if isinstance(metadata_value, Mapping):
        candidate = metadata_value.get("Name")
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown-distribution"


def _entry_point_key(entry_point: object) -> tuple[str, str, str]:
    return (
        str(getattr(entry_point, "name", "")),
        _distribution_name(entry_point),
        str(getattr(entry_point, "value", "")),
    )


def discover_entry_points() -> tuple[metadata.EntryPoint, ...]:
    """Discover metadata only; no discovered implementation is imported."""

    discovered = tuple(metadata.entry_points(group=ENTRY_POINT_GROUP))
    return tuple(sorted(discovered, key=_entry_point_key))


def _load_official_records() -> tuple[Mapping[str, object], ...]:
    registry_file = resources.files("worldzero.laws").joinpath("official_registry.json")
    try:
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialRegistryError("Bundled official law registry cannot be read") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "worldzero-official-law-registry-v1":
        raise OfficialRegistryError("Bundled official law registry has an invalid schema")
    rows = payload.get("approved_families")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise OfficialRegistryError("Bundled official law registry has invalid approved_families")
    return tuple(rows)


def _index_official_records(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    required = {
        "family_id",
        "api_version",
        "family_version",
        "package",
        "package_version",
        "fingerprint",
        "calibration_suite_sha256",
        "release_status",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise OfficialRegistryError("Official registry row has invalid fields")
        family_id = row["family_id"]
        try:
            _require_family_id(family_id)
        except ValueError as exc:
            raise OfficialRegistryError("Official registry row has invalid family_id") from exc
        if family_id in result:
            raise OfficialRegistryError(f"Official registry contains duplicate row for {family_id}")
        for field_name in (
            "api_version",
            "family_version",
            "package",
            "package_version",
            "fingerprint",
            "calibration_suite_sha256",
            "release_status",
        ):
            if not isinstance(row[field_name], str) or not row[field_name]:
                raise OfficialRegistryError(f"Official registry {field_name} is invalid")
        for digest_name in ("fingerprint", "calibration_suite_sha256"):
            digest = row[digest_name]
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise OfficialRegistryError(f"Official registry {digest_name} is invalid")
        if row["release_status"] != "approved":
            raise OfficialRegistryError("Official registry release_status must be approved")
        result[family_id] = dict(row)
    return result


class LawRegistry:
    """Process-local exact-ID registry with lazy installed-plugin loading."""

    def __init__(
        self,
        *,
        builtins: Iterable[LawFamily] = (),
        entry_points: Iterable[metadata.EntryPoint] = (),
        official_records: Iterable[Mapping[str, object]] | None = None,
    ) -> None:
        self._builtins: dict[str, LawFamily] = {}
        self._entry_points: dict[str, metadata.EntryPoint] = {}
        self._resolved: dict[str, RegisteredFamily] = {}
        rows = _load_official_records() if official_records is None else tuple(official_records)
        self._official = _index_official_records(rows)

        for family in builtins:
            self._add_builtin(family)
        for entry_point in entry_points:
            name = getattr(entry_point, "name", None)
            try:
                _require_family_id(name)
            except ValueError as exc:
                raise LawRegistryError("Entry-point name must be an exact family ID") from exc
            if name in self._builtins or name in self._entry_points:
                raise DuplicateFamilyError(f"Duplicate law family ID: {name}")
            self._entry_points[name] = entry_point

    def _add_builtin(self, family: LawFamily) -> None:
        value = _validate_family(family)
        family_id = value.descriptor.family_id
        if family_id in self._builtins or family_id in self._entry_points:
            raise DuplicateFamilyError(f"Duplicate law family ID: {family_id}")
        self._builtins[family_id] = value

    def register_builtin(self, family: LawFamily) -> None:
        """Register a built-in without consulting package metadata."""

        self._add_builtin(family)

    def list_family_ids(self) -> tuple[str, ...]:
        """List advertised exact IDs without importing entry points."""

        return tuple(sorted((*self._builtins, *self._entry_points)))

    def list_ids(self) -> tuple[str, ...]:
        """Compatibility spelling for deterministic ID listing."""

        return self.list_family_ids()

    def _load_entry_point(self, family_id: str) -> LawFamily:
        entry_point = self._entry_points[family_id]
        distribution = _distribution_name(entry_point)
        identity = f"distribution {distribution!r}, entry point {family_id!r}"
        try:
            loaded = entry_point.load()
        except Exception as exc:
            raise FamilyLoadError(f"Failed to load {identity}: {type(exc).__name__}: {exc}") from exc
        if isinstance(loaded, LawFamily):
            family = loaded
        elif callable(loaded):
            try:
                candidate = loaded()
            except Exception as exc:
                raise FamilyLoadError(
                    f"Failed to call zero-argument factory from {identity}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(candidate, LawFamily):
                raise FamilyLoadError(f"Factory from {identity} did not return a LawFamily instance")
            family = candidate
        else:
            raise FamilyLoadError(f"{identity} did not expose a LawFamily instance or factory")
        try:
            _validate_family(family)
        except (TypeError, ValueError) as exc:
            raise FamilyLoadError(f"Invalid LawFamily from {identity}: {exc}") from exc
        if family.descriptor.family_id != family_id:
            raise FamilyLoadError(
                f"LawFamily descriptor ID {family.descriptor.family_id!r} does not match "
                f"entry point {family_id!r} from distribution {distribution!r}"
            )
        return family

    def _official_status(self, family: LawFamily, fingerprint: str) -> bool:
        row = self._official.get(family.descriptor.family_id)
        if row is None:
            return False
        descriptor = family.descriptor
        actual = {
            "api_version": descriptor.api_version,
            "family_version": descriptor.family_version,
            "package": descriptor.package,
            "package_version": descriptor.package_version,
            "fingerprint": fingerprint,
        }
        drift = [name for name, value in actual.items() if row[name] != value]
        if drift:
            raise OfficialRegistryError(
                f"Official family {descriptor.family_id} has registry drift in: {', '.join(drift)}"
            )
        calibration_fingerprint = calibration_suite_fingerprint(family)
        if row["calibration_suite_sha256"] != calibration_fingerprint:
            raise OfficialRegistryError(
                f"Official family {descriptor.family_id} has calibration suite drift"
            )
        return True

    def resolve(self, family_id: str) -> RegisteredFamily:
        """Resolve one exact ID and import only its selected entry point."""

        try:
            _require_family_id(family_id)
        except ValueError as exc:
            raise FamilyNotFoundError(f"No registration for exact family ID {family_id!r}") from exc
        cached = self._resolved.get(family_id)
        if cached is not None:
            return cached
        if family_id in self._builtins:
            family = self._builtins[family_id]
            origin = "builtin"
        elif family_id in self._entry_points:
            family = self._load_entry_point(family_id)
            origin = f"entry-point:{_distribution_name(self._entry_points[family_id])}:{family_id}"
        else:
            raise FamilyNotFoundError(f"No registration for exact family ID {family_id!r}")
        try:
            fingerprint = fingerprint_family(family)
        except (TypeError, ValueError, FamilyLoadError) as exc:
            if family_id in self._entry_points and not isinstance(exc, FamilyLoadError):
                raise FamilyLoadError(f"Failed to fingerprint selected entry point {family_id!r}: {exc}") from exc
            raise
        registered = RegisteredFamily(
            family=family,
            origin=origin,
            official=self._official_status(family, fingerprint),
            fingerprint=fingerprint,
        )
        self._resolved[family_id] = registered
        return registered

    def validate_all(self) -> tuple[RegisteredFamily, ...]:
        """Explicitly load and validate all advertised families in ID order."""

        return tuple(self.resolve(family_id) for family_id in self.list_family_ids())


def builtin_registry() -> LawRegistry:
    """Return the built-in-only registry without package metadata discovery."""

    from .builtin import builtin_families

    return LawRegistry(builtins=builtin_families())


def installed_registry() -> LawRegistry:
    """Return built-ins plus installed metadata without importing plugin code."""

    from .builtin import builtin_families

    return LawRegistry(
        builtins=builtin_families(),
        entry_points=discover_entry_points(),
    )


def resolve_family(family_id: str, *, registry: LawRegistry | None = None) -> RegisteredFamily:
    """Resolve an exact family ID, discovering installed metadata only if needed."""

    if registry is not None:
        return registry.resolve(family_id)
    builtins = builtin_registry()
    if family_id in builtins.list_family_ids():
        return builtins.resolve(family_id)
    discovered = LawRegistry(entry_points=discover_entry_points())
    return discovered.resolve(family_id)


__all__ = [
    "ENTRY_POINT_GROUP",
    "DuplicateFamilyError",
    "FamilyLoadError",
    "FamilyNotFoundError",
    "LawRegistry",
    "LawRegistryError",
    "OfficialRegistryError",
    "RegisteredFamily",
    "builtin_registry",
    "calibration_suite_fingerprint",
    "discover_entry_points",
    "fingerprint_family",
    "installed_registry",
    "resolve_family",
]
