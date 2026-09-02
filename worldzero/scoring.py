"""Central, family-independent scoring profiles and evidence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from types import MappingProxyType

from .laws.registry import LawRegistry, calibration_suite_fingerprint, builtin_registry
from .laws.types import FamilyEvidence


def _freeze_thresholds(value: object, *, path: str = "thresholds") -> object:
    if value is None or isinstance(value, (str, bool)) or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _freeze_thresholds(value[key], path=f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_thresholds(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must be finite JSON")


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ScoringProfile:
    profile_id: str
    version: str
    thresholds: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a nonempty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a nonempty string")
        frozen = _freeze_thresholds(self.thresholds)
        if not isinstance(frozen, Mapping):
            raise TypeError("thresholds must be a JSON object")
        object.__setattr__(self, "thresholds", frozen)

    def persistence_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "thresholds": _thaw(self.thresholds),
        }

    def identity_dict(self) -> dict[str, object]:
        persisted = self.persistence_dict()
        encoded = json.dumps(
            persisted["thresholds"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "thresholds_sha256": hashlib.sha256(encoded).hexdigest(),
        }


def default_scoring_profile() -> ScoringProfile:
    return ScoringProfile(
        "worldzero:mechanical-screen",
        "1.0.0",
        {
            "active_completed_required": 8,
            "active_functional_required": 4,
            "active_requested_required": 8,
            "active_advantage_min": 0.25,
            "active_invalid_action_rate_max": 0.05,
            "null_false_positives_max": 0,
        },
    )


_THRESHOLD_FIELDS = {
    "active_completed_required",
    "active_functional_required",
    "active_requested_required",
    "active_advantage_min",
    "active_invalid_action_rate_max",
    "null_false_positives_max",
}
_ROW_FIELDS = {
    "seed", "arm", "status", "censor_reason", "decisions", "invalid_actions",
    "evidence",
}
_MATCHED_CONTROL_FIELDS = {"family_identity", "control_assignment"}
_LEGACY_FAMILY_IDENTITY_FIELDS = {
    "family_id", "family_version", "fingerprint", "calibration_suite_sha256",
}
_FULL_FAMILY_IDENTITY_FIELDS = {
    "descriptor", "fingerprint", "calibration_suite_sha256", "origin",
    "official", "experimental", "scoring_profile",
}
_SCORING_IDENTITY_FIELDS = {"profile_id", "version", "thresholds_sha256"}
_ARMS = ("active", "forager", "null")
_EPISODE_CENSOR_REASONS = frozenset({"model_call_budget", "decision_budget"})
_INHERITANCE_CENSOR_REASONS = frozenset({"successor_censoring"})


def _validated_thresholds(profile: ScoringProfile) -> dict[str, int | float]:
    if not isinstance(profile, ScoringProfile) or set(profile.thresholds) != _THRESHOLD_FIELDS:
        raise ValueError("Scoring profile thresholds do not match the central scoring contract")
    values = dict(profile.thresholds)
    for field in (
        "active_completed_required", "active_functional_required",
        "active_requested_required", "null_false_positives_max",
    ):
        if type(values[field]) is not int or values[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    for field in ("active_advantage_min", "active_invalid_action_rate_max"):
        if type(values[field]) not in (int, float) or not 0 <= values[field] <= 1:
            raise ValueError(f"{field} must be a rate in [0, 1]")
    if values["active_requested_required"] <= 0:
        raise ValueError("active_requested_required must be positive")
    if values["active_completed_required"] > values["active_requested_required"]:
        raise ValueError("active_completed_required exceeds requested worlds")
    if values["active_functional_required"] > values["active_requested_required"]:
        raise ValueError("active_functional_required exceeds requested worlds")
    return values  # type: ignore[return-value]


def _validated_evidence(value: object) -> FamilyEvidence:
    if isinstance(value, FamilyEvidence):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("evidence must be FamilyEvidence or its persistence mapping")
    return FamilyEvidence.from_persistence(value)  # type: ignore[arg-type]


def _validated_outcome(value: object, *, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "status", "censor_reason", "survived", "age",
    }:
        raise ValueError(f"{path} successor outcome fields are invalid")
    status = value["status"]
    if status not in {"completed", "censored"}:
        raise ValueError(f"{path} successor status is invalid")
    reason = value["censor_reason"]
    survived = value["survived"]
    age = value["age"]
    if status == "completed":
        if reason is not None:
            raise ValueError(f"{path} completed successor cannot have a censor reason")
        if type(survived) is not bool:
            raise TypeError(f"{path} completed successor survived must be boolean")
        if type(age) not in (int, float) or not math.isfinite(age) or age < 0:
            raise ValueError(f"{path} successor age must be finite and nonnegative")
    else:
        if reason not in _EPISODE_CENSOR_REASONS:
            raise ValueError(f"{path} censored successor has an invalid censor reason")
        if survived is not None:
            raise ValueError(f"{path} censored successor survived must be null")
        if age is not None and (
            type(age) not in (int, float) or not math.isfinite(age) or age < 0
        ):
            raise ValueError(f"{path} censored successor age must be null or finite")
    return {"status": status, "censor_reason": reason, "survived": survived, "age": age}


def _validated_inheritance(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "status", "censor_reason", "eligible", "retained", "knockout", "broken",
    }:
        raise ValueError("inheritance outcome fields are invalid")
    status = value["status"]
    if status not in {"completed", "censored", "not_selected"}:
        raise ValueError("inheritance status is invalid")
    if type(value["eligible"]) is not bool:
        raise TypeError("inheritance eligible must be boolean")
    reason = value["censor_reason"]
    branches: dict[str, object] = {}
    for branch in ("retained", "knockout", "broken"):
        branch_value = value[branch]
        if status in {"completed", "censored"}:
            branches[branch] = _validated_outcome(branch_value, path=branch)
        elif branch_value is not None:
            raise ValueError("not-selected inheritance branches must be null")
        else:
            branches[branch] = None
    branch_statuses = [
        branches[branch]["status"]
        for branch in ("retained", "knockout", "broken")
        if isinstance(branches[branch], dict)
    ]
    if status == "completed":
        if reason is not None:
            raise ValueError("completed inheritance cannot have a censor reason")
        if any(branch_status != "completed" for branch_status in branch_statuses):
            raise ValueError("completed inheritance cannot contain a censored branch")
    elif status == "censored":
        if reason not in _INHERITANCE_CENSOR_REASONS:
            raise ValueError("censored inheritance has an invalid censor reason")
        if "censored" not in branch_statuses:
            raise ValueError("censored inheritance must contain a censored branch")
    elif reason is not None:
        raise ValueError("not-selected inheritance cannot have a censor reason")
    return {
        "status": status, "censor_reason": reason,
        "eligible": value["eligible"], **branches,
    }


def _validated_rows(
    rows: Sequence[Mapping[str, object]], required: int,
) -> list[dict[str, object]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a detached sequence of mappings")
    result: list[dict[str, object]] = []
    matched_flags = [
        isinstance(raw, Mapping)
        and bool(set(raw) & _MATCHED_CONTROL_FIELDS)
        for raw in rows
    ]
    if any(matched_flags) and not all(matched_flags):
        raise ValueError("mixed legacy and matched-control scoring rows are invalid")
    matched_controls = bool(matched_flags and all(matched_flags))
    seen: set[tuple[int, str]] = set()
    seeds_by_arm: dict[str, set[int]] = {arm: set() for arm in _ARMS}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"rows[{index}] must be a mapping")
        required_fields = _ROW_FIELDS | (
            _MATCHED_CONTROL_FIELDS if matched_controls else set()
        )
        allowed = required_fields | {"inheritance"}
        if not required_fields <= set(raw) or not set(raw) <= allowed:
            raise ValueError(f"rows[{index}] contains missing or unknown denominator fields")
        seed, arm, status = raw["seed"], raw["arm"], raw["status"]
        if type(seed) is not int or seed < 0:
            raise ValueError("row seed must be a nonnegative integer")
        if arm not in _ARMS:
            raise ValueError("row arm is invalid")
        if status not in {"completed", "censored"}:
            raise ValueError("row status is invalid")
        censor_reason = raw["censor_reason"]
        if status == "completed" and censor_reason is not None:
            raise ValueError("completed row cannot have a censor reason")
        if status == "censored" and censor_reason not in _EPISODE_CENSOR_REASONS:
            raise ValueError("censored row has an invalid censor reason")
        identity = (seed, arm)
        if identity in seen:
            raise ValueError("duplicate seed+arm row")
        seen.add(identity)
        seeds_by_arm[arm].add(seed)
        decisions, invalid = raw["decisions"], raw["invalid_actions"]
        if type(decisions) is not int or decisions < 0:
            raise ValueError("decisions must be a nonnegative integer")
        if type(invalid) is not int or invalid < 0 or invalid > decisions:
            raise ValueError("invalid_actions is inconsistent with decisions")
        inherited = _validated_inheritance(raw.get("inheritance"))
        if (status == "censored" and isinstance(inherited, dict)
                and inherited["status"] != "not_selected"):
            raise ValueError("censored ancestor cannot have inheritance outcomes")
        validated_row = {
            "seed": seed,
            "arm": arm,
            "status": status,
            "censor_reason": censor_reason,
            "decisions": decisions,
            "invalid_actions": invalid,
            "evidence": _validated_evidence(raw["evidence"]),
            "inheritance": inherited,
        }
        if matched_controls:
            identity_value = raw["family_identity"]
            if not isinstance(identity_value, Mapping):
                raise ValueError("matched-control family identity is invalid")
            fields = set(identity_value)
            if fields == _LEGACY_FAMILY_IDENTITY_FIELDS:
                if (
                    any(not isinstance(identity_value[field], str)
                        for field in _LEGACY_FAMILY_IDENTITY_FIELDS)
                    or not _valid_sha256(identity_value["fingerprint"])
                    or not _valid_sha256(identity_value["calibration_suite_sha256"])
                ):
                    raise ValueError("matched-control family identity is invalid")
            elif fields == _FULL_FAMILY_IDENTITY_FIELDS:
                if not _valid_full_identity_shape(identity_value):
                    raise ValueError("matched-control family identity is invalid")
            else:
                raise ValueError("matched-control family identity is invalid")
            assignment = raw["control_assignment"]
            expected_assignment = "matched_null" if arm == "null" else "active"
            if assignment != expected_assignment:
                raise ValueError("matched-control control assignment is invalid")
            validated_row["family_identity"] = dict(identity_value)
            validated_row["control_assignment"] = assignment
        result.append(validated_row)
    if any(len(seeds_by_arm[arm]) != required for arm in _ARMS):
        raise ValueError("missing active or baseline coverage")
    if not (seeds_by_arm["active"] == seeds_by_arm["forager"] == seeds_by_arm["null"]):
        raise ValueError("active and baseline seed coverage is mismatched")
    if matched_controls:
        identities = [row["family_identity"] for row in result]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError("matched-control family identity is mismatched")
    return result


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_full_identity_shape(identity: Mapping[str, object]) -> bool:
    descriptor = identity.get("descriptor")
    scoring_identity = identity.get("scoring_profile")
    return (
        isinstance(descriptor, Mapping)
        and isinstance(descriptor.get("family_id"), str)
        and isinstance(identity.get("origin"), str)
        and bool(identity.get("origin"))
        and type(identity.get("official")) is bool
        and type(identity.get("experimental")) is bool
        and identity.get("experimental") is (not identity.get("official"))
        and _valid_sha256(identity.get("fingerprint"))
        and _valid_sha256(identity.get("calibration_suite_sha256"))
        and isinstance(scoring_identity, Mapping)
        and set(scoring_identity) == _SCORING_IDENTITY_FIELDS
        and isinstance(scoring_identity.get("profile_id"), str)
        and isinstance(scoring_identity.get("version"), str)
        and _valid_sha256(scoring_identity.get("thresholds_sha256"))
    )


def _validate_registered_scoring_identity(
    identity: Mapping[str, object],
    profile: ScoringProfile,
    *,
    registry: LawRegistry | None,
    experimental_family: bool,
) -> None:
    if identity["scoring_profile"] != profile.identity_dict():
        raise ValueError("matched-control scoring profile identity drift")
    resolving_registry = builtin_registry() if registry is None else registry
    if not isinstance(resolving_registry, LawRegistry):
        raise TypeError("registry must be a LawRegistry")
    descriptor = identity["descriptor"]
    assert isinstance(descriptor, Mapping)
    family_id = descriptor["family_id"]
    assert isinstance(family_id, str)
    try:
        registered = resolving_registry.resolve(family_id)
    except Exception as exc:
        raise ValueError("official scoring could not resolve the exact family identity") from exc
    runtime_identity = {
        "descriptor": registered.family.descriptor.persistence_dict(),
        "fingerprint": registered.fingerprint,
        "calibration_suite_sha256": calibration_suite_fingerprint(registered.family),
        "origin": registered.origin,
        "scoring_profile": profile.identity_dict(),
    }
    if experimental_family:
        expected = {
            **runtime_identity,
            "official": False,
            "experimental": True,
        }
        if dict(identity) != expected:
            raise ValueError("matched-control registered family identity drift")
        try:
            bundled = builtin_registry().resolve(family_id)
        except Exception:
            return
        bundled_identity = {
            "descriptor": bundled.family.descriptor.persistence_dict(),
            "fingerprint": bundled.fingerprint,
            "calibration_suite_sha256": calibration_suite_fingerprint(bundled.family),
            "origin": bundled.origin,
            "scoring_profile": profile.identity_dict(),
        }
        if runtime_identity == bundled_identity:
            raise ValueError("experimental scoring refuses an exact official family identity")
        return

    try:
        bundled = builtin_registry().resolve(family_id)
    except Exception as exc:
        raise ValueError(
            "official scoring refuses a family outside the bundled official registry"
        ) from exc
    expected = {
        "descriptor": bundled.family.descriptor.persistence_dict(),
        "fingerprint": bundled.fingerprint,
        "calibration_suite_sha256": calibration_suite_fingerprint(bundled.family),
        "origin": bundled.origin,
        "official": True,
        "experimental": False,
        "scoring_profile": profile.identity_dict(),
    }
    runtime_official_identity = {
        **runtime_identity,
        "official": True,
        "experimental": False,
    }
    if runtime_official_identity != expected or dict(identity) != expected:
        raise ValueError("official scoring registered family identity drift")


def _functional(evidence: FamilyEvidence) -> bool:
    return (
        evidence.structure_constructed
        and evidence.function_observed
        and evidence.effect_observed
    )


def _inheritance_summary(active_rows: list[dict[str, object]]) -> dict[str, object]:
    completed_ancestors = [row for row in active_rows if row["status"] == "completed"]
    completed_outcomes: list[dict[str, object]] = []
    eligible_outcomes: list[dict[str, object]] = []
    eligible_ancestors = 0
    inheritance_counts = Counter()
    branch_censor_counts = Counter({branch: 0 for branch in ("retained", "knockout", "broken")})
    censor_records: list[dict[str, object]] = []
    for row in completed_ancestors:
        inherited = row["inheritance"]
        if not isinstance(inherited, dict):
            inheritance_counts["not_reported"] += 1
            continue
        inheritance_counts[inherited["status"]] += 1
        if isinstance(inherited, dict) and inherited["eligible"] is True:
            eligible_ancestors += 1
        if inherited["status"] == "censored":
            branch_reasons = {}
            for branch in ("retained", "knockout", "broken"):
                outcome = inherited[branch]
                if isinstance(outcome, dict) and outcome["status"] == "censored":
                    branch_censor_counts[branch] += 1
                    branch_reasons[branch] = outcome["censor_reason"]
            censor_records.append({
                "seed": row["seed"],
                "censor_reason": inherited["censor_reason"],
                "branches": branch_reasons,
                "eligible": inherited["eligible"],
            })
            continue
        if inherited["status"] != "completed":
            continue
        branches = [inherited[name] for name in ("retained", "knockout", "broken")]
        if any(not isinstance(branch, dict) or branch["status"] != "completed" for branch in branches):
            continue
        if inherited["eligible"] is True:
            completed_outcomes.append(inherited)
            eligible_outcomes.append(inherited)

    def effect(rows: list[dict[str, object]], denominator: int, other: str) -> dict[str, object]:
        numerator = sum(
            int(row["retained"]["survived"]) - int(row[other]["survived"])
            for row in rows
        )
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator if denominator else None,
        }

    all_n = len(completed_ancestors)
    eligible_n = len(eligible_outcomes)
    censored_n = inheritance_counts["censored"]
    eligible_censored_n = sum(record["eligible"] is True for record in censor_records)
    common = {
        "eligible_ancestors": eligible_ancestors,
        "n_successor_outcomes_censored": censored_n,
        "branch_censor_counts": {
            branch: branch_censor_counts[branch]
            for branch in ("retained", "knockout", "broken")
        },
        "censor_reasons": censor_records,
        "assignment_counts": {
            status: inheritance_counts[status]
            for status in ("completed", "censored", "not_selected", "not_reported")
        },
    }
    return {
        "all_completed_ancestors": {
            "denominator": all_n,
            "n_successor_outcomes_completed": len(completed_outcomes),
            "n_unselected_or_ineligible": all_n - len(completed_outcomes),
            "mechanism_effect": effect(completed_outcomes, all_n, "knockout"),
            "broken_relation_effect": effect(completed_outcomes, all_n, "broken"),
            "assignment": "unselected and incomplete successor outcomes contribute zero",
            **common,
        },
        "eligible_only": {
            "denominator": eligible_n,
            "eligible_ancestors": eligible_ancestors,
            "n_successor_outcomes_completed": eligible_n,
            "n_successor_outcomes_censored": eligible_censored_n,
            "branch_censor_counts": {
                branch: sum(
                    record["eligible"] is True and branch in record["branches"]
                    for record in censor_records
                )
                for branch in ("retained", "knockout", "broken")
            },
            "censor_reasons": [
                record for record in censor_records if record["eligible"] is True
            ],
            "mechanism_effect": effect(eligible_outcomes, eligible_n, "knockout"),
            "broken_relation_effect": effect(eligible_outcomes, eligible_n, "broken"),
        },
    }


def score_evidence(
    rows: Sequence[Mapping[str, object]], profile: ScoringProfile, *,
    registry: LawRegistry | None = None,
    experimental_family: bool = False,
) -> dict[str, object]:
    """Apply one frozen central profile to detached standardized evidence rows."""

    if type(experimental_family) is not bool:
        raise TypeError("experimental_family must be a boolean")
    thresholds = _validated_thresholds(profile)
    requested = int(thresholds["active_requested_required"])
    validated = _validated_rows(rows, requested)
    matched_controls = "family_identity" in validated[0]
    full_identity = (
        matched_controls
        and set(validated[0]["family_identity"]) == _FULL_FAMILY_IDENTITY_FIELDS
    )
    if full_identity:
        _validate_registered_scoring_identity(
            validated[0]["family_identity"],  # type: ignore[arg-type]
            profile,
            registry=registry,
            experimental_family=experimental_family,
        )
    elif experimental_family:
        raise ValueError(
            "experimental scoring requires the full registered family identity"
        )
    by_arm = {
        arm: [row for row in validated if row["arm"] == arm]
        for arm in _ARMS
    }
    active = by_arm["active"]
    forager = by_arm["forager"]
    null = by_arm["null"]
    completed = {arm: sum(row["status"] == "completed" for row in by_arm[arm]) for arm in _ARMS}
    active_functional = sum(
        row["status"] == "completed" and _functional(row["evidence"])
        for row in active
    )
    forager_functional = sum(
        row["status"] == "completed" and _functional(row["evidence"])
        for row in forager
    )
    active_rate = active_functional / requested
    forager_rate = forager_functional / requested
    advantage = active_rate - forager_rate
    invalid_numerator = sum(int(row["invalid_actions"]) for row in active)
    invalid_denominator = sum(int(row["decisions"]) for row in active)
    invalid_rate = invalid_numerator / invalid_denominator if invalid_denominator else None
    null_false_positives = sum(
        row["status"] == "completed"
        and (row["evidence"].function_observed or row["evidence"].effect_observed)
        for row in null
    )
    metrics = {
        "active_functional": {
            "numerator": active_functional,
            "denominator": requested,
            "rate": active_rate,
            "threshold": thresholds["active_functional_required"],
            "passed": active_functional >= thresholds["active_functional_required"],
        },
        "active_advantage": {
            "active_numerator": active_functional,
            "forager_numerator": forager_functional,
            "denominator": requested,
            "active_rate": active_rate,
            "forager_rate": forager_rate,
            "rate": advantage,
            "threshold": thresholds["active_advantage_min"],
            "passed": advantage >= thresholds["active_advantage_min"],
        },
        "active_invalid_actions": {
            "numerator": invalid_numerator,
            "denominator": invalid_denominator,
            "rate": invalid_rate,
            "threshold": thresholds["active_invalid_action_rate_max"],
            "passed": invalid_rate is not None and invalid_rate <= thresholds["active_invalid_action_rate_max"],
        },
        "null_false_positives": {
            "numerator": null_false_positives,
            "denominator": requested,
            "rate": null_false_positives / requested,
            "threshold": thresholds["null_false_positives_max"],
            "passed": null_false_positives <= thresholds["null_false_positives_max"],
        },
    }
    coverage_checks = {
        "active_requested": {
            "numerator": len(active), "denominator": requested,
            "threshold": thresholds["active_requested_required"],
            "passed": len(active) == thresholds["active_requested_required"],
        },
        "active_completed": {
            "numerator": completed["active"], "denominator": requested,
            "threshold": thresholds["active_completed_required"],
            "passed": completed["active"] == thresholds["active_completed_required"],
        },
        "matched_forager": {
            "numerator": len(forager), "denominator": requested,
            "threshold": requested, "passed": len(forager) == requested,
        },
        "matched_null": {
            "numerator": len(null), "denominator": requested,
            "threshold": requested, "passed": len(null) == requested,
        },
    }
    origin_counts = Counter(row["evidence"].origin for row in active)
    censor_reasons = [
        {
            "seed": row["seed"], "arm": row["arm"], "status": row["status"],
            "censor_reason": row["censor_reason"],
        }
        for row in validated if row["status"] != "completed"
    ]
    arm_order = {arm: index for index, arm in enumerate(_ARMS)}
    censor_reasons.sort(key=lambda row: (arm_order[row["arm"]], row["seed"]))
    all_passed = all(item["passed"] for item in (*coverage_checks.values(), *metrics.values()))
    decision = (
        "INCOMPLETE_CENSORED" if censor_reasons else
        "WORTH_INVESTIGATING" if all_passed else
        "NOT_WORTH_INVESTIGATING"
    )
    result = {
        "schema": (
            "worldzero-central-score-v3"
            if full_identity else
            "worldzero-central-score-v2"
            if matched_controls else
            "worldzero-central-score-v1"
        ),
        "profile": {
            **profile.persistence_dict(),
            "thresholds_sha256": profile.identity_dict()["thresholds_sha256"],
        },
        "coverage": {
            "arms": {
                arm: {"requested": len(by_arm[arm]), "completed": completed[arm],
                      "censored": len(by_arm[arm]) - completed[arm]}
                for arm in _ARMS
            },
            "checks": coverage_checks,
        },
        "metrics": metrics,
        "censor_reasons": censor_reasons,
        "origin_counts": {
            origin: origin_counts.get(origin, 0)
            for origin in ("model_placement", "model_drop", "death_drop", "pre_existing", "none")
        },
        "inheritance": _inheritance_summary(active),
        "decision": decision,
        "interpretation": (
            "WORTH_INVESTIGATING is not discovery proof; standardized evidence and "
            "matched controls require independent review."
        ),
    }
    if matched_controls:
        result["family_identity"] = copy.deepcopy(validated[0]["family_identity"])
        result["control_contract"] = {
            "active": "active", "forager": "active", "null": "matched_null",
        }
    return result


__all__ = ["ScoringProfile", "default_scoring_profile", "score_evidence"]
