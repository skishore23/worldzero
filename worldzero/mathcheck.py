"""Independent analytic checks against actual production-world trajectories."""
import copy
from dataclasses import replace
import math
from typing import Any
import numpy as np
from .kernel import Config, Law, World, encode_transition_operation
from .laws.base import LawFamily
from .laws.builtin import builtin_families
from .laws.registry import calibration_suite_fingerprint, fingerprint_family
from .laws.types import (
    AccountingDelta,
    CalibrationCase,
    EventEvidence,
    InterventionTransition,
    KernelProposalRejection,
    LawTransition,
    ModulePositionChange,
    PrivateStateTransition,
    ProposalDraw,
    ResourcePreservation,
    SampleContext,
    SubstrateView,
    TargetDomain,
    thaw_json,
)


# The benchmark, rather than a plugin, owns the exhaustive calibration surface.
# Exact metadata is intentional: changing a tolerance, sample count, or declared
# expectation is a benchmark-version change, not a family-local escape hatch.
_OFFICIAL_CALIBRATION_SUITES: dict[str, tuple[dict[str, object], ...]] = {
    "worldzero:catalysis": (
        {"case_id": "legacy-conversion-rate", "kind": "analytic", "expected": 0.45,
         "absolute_tolerance": 0.0, "relative_tolerance": 0.0, "samples": 1,
         "parameters": {"channel": "convert"}},
        {"case_id": "conversion-accounting", "kind": "invariant",
         "expected": {"energy_delta": 5.2, "material_delta": 0},
         "absolute_tolerance": 0.0, "relative_tolerance": 0.0, "samples": 1,
         "parameters": {}},
    ),
    "worldzero:delayed-transformation": (
        {"case_id": "exact-dwell-boundary", "kind": "analytic", "expected": True,
         "absolute_tolerance": 0.0, "relative_tolerance": 0.0, "samples": 1,
         "parameters": {"clock": "simulated_time"}},
        {"case_id": "break-resets-maturation", "kind": "invariant", "expected": True,
         "absolute_tolerance": 0.0, "relative_tolerance": 0.0, "samples": 1,
         "parameters": {}},
    ),
    "worldzero:inhibition": (
        {"case_id": "inside-field-raw-decay-rejection", "kind": "invariant",
         "expected": True, "absolute_tolerance": 0.0, "relative_tolerance": 0.0,
         "samples": 1, "parameters": {"accounting_delta": [0, 0.0]}},
        {"case_id": "outside-field-matched-decay", "kind": "invariant",
         "expected": True, "absolute_tolerance": 0.0, "relative_tolerance": 0.0,
         "samples": 1, "parameters": {}},
    ),
    "worldzero:null": (
        {"case_id": "matched-conversion-envelope", "kind": "invariant",
         "expected": True, "absolute_tolerance": 0.0, "relative_tolerance": 0.0,
         "samples": 1,
         "parameters": {"channel": "convert", "mechanism_effect": False}},
    ),
}


def _calibration_case_contract(case: object) -> dict[str, object]:
    return {
        "absolute_tolerance": case.absolute_tolerance,  # type: ignore[attr-defined]
        "case_id": case.case_id,  # type: ignore[attr-defined]
        "expected": thaw_json(case.expected),  # type: ignore[attr-defined]
        "kind": case.kind,  # type: ignore[attr-defined]
        "parameters": thaw_json(case.parameters),  # type: ignore[attr-defined]
        "relative_tolerance": case.relative_tolerance,  # type: ignore[attr-defined]
        "samples": case.samples,  # type: ignore[attr-defined]
    }


def _calibration_fixture(family: LawFamily, *, now: float = 5.0):
    config = Config(width=9, height=7)
    instance = family.sample(SampleContext({
        "dwell_duration": 3.0,
        "legacy_geometry": "adjacent",
        "legacy_pair": [0, 1],
        "module_count": 3,
        "raw_energy": config.raw_energy,
        "resource_energy_gain": config.rich_energy - config.raw_energy,
        "rich_energy": config.rich_energy,
    }))
    target = 2 * config.width + 4
    resources = [[0 for _ in range(config.width)] for _ in range(config.height)]
    resources[2][4] = 1
    view = SubstrateView(
        width=config.width,
        height=config.height,
        agent_position=(3, 4),
        module_positions=((3, 4), (3, 5), None),
        module_states=({}, {}, {}),
        resources=tuple(tuple(row) for row in resources),
        terrain=tuple(tuple(1 for _ in range(config.width)) for _ in range(config.height)),
        simulated_time=now,
        kernel_counters={"assemblies": 0, "conversions": 0, "proposal_count": 4},
    )
    return config, instance, view, target


def _channel_contract(family: LawFamily, instance: object, config: Config) -> list[dict[str, object]]:
    return [
        {
            "channel_id": channel.channel_id,
            "draw_requirements": [item.value for item in channel.draw_requirements],
            "envelope_rate": channel.envelope_rate,
            "target_domain": channel.target_domain.value,
        }
        for channel in family.channels(instance, config)  # type: ignore[arg-type]
    ]


def _operation_contract(operations: tuple[object, ...]) -> list[dict[str, object]]:
    return [encode_transition_operation(operation) for operation in operations]


def _effect_contract(
    transition: LawTransition | KernelProposalRejection | None,
    *, kind: str,
) -> dict[str, object]:
    if transition is None:
        return {
            "accounting": {"energy_delta": 0.0, "material_delta": 0},
            "capabilities": [],
            "evidence": [],
            "kind": "no_op",
            "operations": [],
        }
    operations = tuple(transition.operations)
    return {
        "accounting": {
            "energy_delta": transition.accounting.energy_delta,
            "material_delta": transition.accounting.material_delta,
        },
        "capabilities": sorted(transition.declared_capabilities),
        "evidence": [
            {
                "details": thaw_json(operation.details),
                "event_type": operation.event_type,
                "location_index": operation.location_index,
            }
            for operation in operations
            if isinstance(operation, EventEvidence)
        ],
        "kind": kind,
        "operations": _operation_contract(operations),
    }


def _private_contract(transition: object) -> dict[str, object]:
    if not isinstance(transition, PrivateStateTransition):
        return {
            "capabilities": [], "expected_state": None,
            "kind": "no_op", "replacement_state": None,
        }
    return {
        "capabilities": sorted(transition.declared_capabilities),
        "expected_state": thaw_json(transition.expected_state),
        "kind": "private_state_transition",
        "replacement_state": thaw_json(transition.replacement_state),
    }


def _intervention_contract(transition: object) -> dict[str, object]:
    if not isinstance(transition, InterventionTransition):
        return {
            "accounting": {"energy_delta": 0.0, "material_delta": 0},
            "capabilities": [], "control": None, "kind": "invalid",
            "operations": [],
        }
    return {
        "accounting": {
            "energy_delta": transition.accounting.energy_delta,
            "material_delta": transition.accounting.material_delta,
        },
        "capabilities": sorted(transition.declared_capabilities),
        "control": transition.control.value,
        "kind": "intervention",
        "operations": _operation_contract(tuple(transition.operations)),
    }


def _view_at(view: SubstrateView, time: float, *, positions=None) -> SubstrateView:
    return SubstrateView(
        view.width, view.height, view.agent_position,
        view.module_positions if positions is None else tuple(positions),
        view.module_states, view.resources, view.terrain, time,
        view.kernel_counters,
    )


_CONVERT_CHANNEL = [{
    "channel_id": "convert",
    "draw_requirements": ["target_index", "acceptance_uniform"],
    "envelope_rate": 28.349999999999998,
    "target_domain": "cell",
}]
_ZERO_EFFECT = {
    "accounting": {"energy_delta": 0.0, "material_delta": 0},
    "capabilities": [], "evidence": [], "kind": "no_op", "operations": [],
}
_CONVERSION_EFFECT = {
    "accounting": {"energy_delta": 5.2, "material_delta": 0},
    "capabilities": ["resource_transition"],
    "evidence": [],
    "kind": "accepted",
    "operations": [{
        "cell_index": 22, "expected_value": 1, "replacement_value": 2,
        "type": "resource_replacement",
    }],
}


def _expected_calibration_observation(case_id: str) -> dict[str, object]:
    """Return the benchmark-owned contract for a stable calibration case ID."""

    if case_id == "legacy-conversion-rate":
        return {"channels": copy.deepcopy(_CONVERT_CHANNEL)}
    if case_id == "conversion-accounting":
        return {
            "channels": copy.deepcopy(_CONVERT_CHANNEL),
            "derived": {"affected_contains_target": True, "functional": True},
            "transition": copy.deepcopy(_CONVERSION_EFFECT),
        }
    if case_id == "matched-conversion-envelope":
        return {
            "channels": copy.deepcopy(_CONVERT_CHANNEL),
            "derived": {
                "affected_locations": [], "functional": False, "structural": True,
            },
            "transition": copy.deepcopy(_ZERO_EFFECT),
        }
    if case_id == "inside-field-raw-decay-rejection":
        return {
            "channels": [],
            "derived": {"affected_contains_target": True, "functional": True},
            "proposal": {
                "acceptance_uniform": 0.25, "channel_id": "raw_decay",
                "proposal_index": 5, "simulated_time": 5.0, "target_index": 22,
            },
            "rejection": {
                "accounting": {"energy_delta": 0.0, "material_delta": 0},
                "capabilities": [
                    "event_evidence", "proposal_filter", "resource_preservation",
                ],
                "evidence": [{
                    "details": {"proposal_index": 5},
                    "event_type": "inhibited_proposal", "location_index": 22,
                }],
                "kind": "rejected",
                "operations": [
                    {"cell_index": 22, "expected_value": 1,
                     "type": "resource_preservation"},
                    {"details": {"proposal_index": 5},
                     "event_type": "inhibited_proposal", "location_index": 22,
                     "type": "event_evidence"},
                ],
            },
        }
    if case_id == "outside-field-matched-decay":
        return {
            "channels": [],
            "derived": {"affected_contains_target": False, "functional": True},
            "proposal": {
                "acceptance_uniform": 0.25, "channel_id": "raw_decay",
                "proposal_index": 5, "simulated_time": 5.0, "target_index": 0,
            },
            "rejection": copy.deepcopy(_ZERO_EFFECT),
        }
    private_start = {
        "capabilities": ["private_state_transition"],
        "expected_state": {"assembled_since": None},
        "kind": "private_state_transition",
        "replacement_state": {"assembled_since": 5.0},
    }
    if case_id == "exact-dwell-boundary":
        return {
            "channels": copy.deepcopy(_CONVERT_CHANNEL),
            "function": {
                "after_boundary": True, "at_boundary": True,
                "before_boundary": False,
            },
            "synchronization": private_start,
            "transition": copy.deepcopy(_CONVERSION_EFFECT),
        }
    if case_id == "break-resets-maturation":
        return {
            "broken_functional": False,
            "channels": copy.deepcopy(_CONVERT_CHANNEL),
            "function_after_rebuild": {
                "after_boundary": True, "at_boundary": True,
                "before_boundary": False,
            },
            "intervention": {
                "accounting": {"energy_delta": 0.0, "material_delta": 0},
                "capabilities": ["geometry_control"], "control": "broken",
                "kind": "intervention",
                "operations": [{
                    "expected_position": [3, 4], "module_index": 0,
                    "replacement_position": [0, 4],
                    "type": "module_position_change",
                }],
            },
            "rebuild": {
                "capabilities": ["private_state_transition"],
                "expected_state": {"assembled_since": None},
                "kind": "private_state_transition",
                "replacement_state": {"assembled_since": 9.0},
            },
            "reset": {
                "capabilities": ["private_state_transition"],
                "expected_state": {"assembled_since": 5.0},
                "kind": "private_state_transition",
                "replacement_state": {"assembled_since": None},
            },
            "start": private_start,
        }
    raise ValueError(f"No benchmark-owned calibration contract for {case_id}")


def _calibration_observation(family: LawFamily, case_id: str) -> dict[str, object]:
    """Execute a stable case contract through the supplied family's real callbacks."""

    config, instance, view, target = _calibration_fixture(family)
    channels = _channel_contract(family, instance, config)
    proposal = ProposalDraw("convert", target, 0.25, 5, view.simulated_time)
    if case_id == "legacy-conversion-rate":
        return {"channels": channels}
    if case_id == "conversion-accounting":
        derived = family.derive(view, instance)
        transition = family.apply_proposal(proposal, view, instance, derived)
        return {
            "channels": channels,
            "derived": {
                "affected_contains_target": target in derived.affected_locations,
                "functional": derived.functional,
            },
            "transition": _effect_contract(transition, kind="accepted"),
        }
    if case_id == "matched-conversion-envelope":
        derived = family.derive(view, instance)
        transition = family.apply_proposal(proposal, view, instance, derived)
        return {
            "channels": channels,
            "derived": {
                "affected_locations": list(derived.affected_locations),
                "functional": derived.functional,
                "structural": derived.state.get("structural"),
            },
            "transition": _effect_contract(transition, kind="accepted"),
        }
    if case_id == "inside-field-raw-decay-rejection":
        derived = family.derive(view, instance)
        decay = ProposalDraw("raw_decay", target, 0.25, 5, view.simulated_time)
        rejection = family.filter_kernel_proposal(decay, view, instance, derived)
        return {
            "channels": channels,
            "derived": {
                "affected_contains_target": target in derived.affected_locations,
                "functional": derived.functional,
            },
            "proposal": {
                "acceptance_uniform": decay.acceptance_uniform,
                "channel_id": decay.channel_id,
                "proposal_index": decay.proposal_index,
                "simulated_time": decay.simulated_time,
                "target_index": decay.target_index,
            },
            "rejection": _effect_contract(rejection, kind="rejected"),
        }
    if case_id == "outside-field-matched-decay":
        outside = 0
        resources = [list(row) for row in view.resources]
        resources[0][0] = 1
        outside_view = SubstrateView(
            view.width, view.height, view.agent_position, view.module_positions,
            view.module_states, tuple(tuple(row) for row in resources), view.terrain,
            view.simulated_time, view.kernel_counters,
        )
        derived = family.derive(outside_view, instance)
        decay = ProposalDraw("raw_decay", outside, 0.25, 5, view.simulated_time)
        rejection = family.filter_kernel_proposal(decay, outside_view, instance, derived)
        return {
            "channels": channels,
            "derived": {
                "affected_contains_target": outside in derived.affected_locations,
                "functional": derived.functional,
            },
            "proposal": {
                "acceptance_uniform": decay.acceptance_uniform,
                "channel_id": decay.channel_id,
                "proposal_index": decay.proposal_index,
                "simulated_time": decay.simulated_time,
                "target_index": decay.target_index,
            },
            "rejection": _effect_contract(rejection, kind="rejected"),
        }
    if case_id == "exact-dwell-boundary":
        started = family.synchronize_private_state(view, instance)
        started_instance = (
            started.resulting_instance(instance)
            if isinstance(started, PrivateStateTransition) else instance
        )
        boundary = view.simulated_time + 3.0
        before_view = _view_at(view, math.nextafter(boundary, -math.inf))
        boundary_view = _view_at(view, boundary)
        after_view = _view_at(view, math.nextafter(boundary, math.inf))
        before = family.derive(before_view, started_instance)
        exact = family.derive(boundary_view, started_instance)
        after = family.derive(after_view, started_instance)
        transition = family.apply_proposal(
            ProposalDraw("convert", target, 0.25, 5, boundary),
            boundary_view, started_instance, exact,
        )
        return {
            "channels": channels,
            "function": {
                "at_boundary": exact.functional,
                "after_boundary": after.functional,
                "before_boundary": before.functional,
            },
            "synchronization": _private_contract(started),
            "transition": _effect_contract(transition, kind="accepted"),
        }
    if case_id == "break-resets-maturation":
        started = family.synchronize_private_state(view, instance)
        started_instance = (
            started.resulting_instance(instance)
            if isinstance(started, PrivateStateTransition) else instance
        )
        intervention = family.intervene("broken", view, started_instance)  # type: ignore[arg-type]
        changes = (
            [operation for operation in intervention.operations
             if isinstance(operation, ModulePositionChange)]
            if isinstance(intervention, InterventionTransition) else []
        )
        positions = list(view.module_positions)
        if len(changes) == 1:
            positions[changes[0].module_index] = changes[0].replacement_position
        broken_view = _view_at(view, view.simulated_time, positions=positions)
        reset = family.synchronize_private_state(broken_view, started_instance)
        reset_instance = (
            reset.resulting_instance(started_instance)
            if isinstance(reset, PrivateStateTransition) else started_instance
        )
        rebuild_time = 9.0
        rebuilt_view = _view_at(view, rebuild_time)
        rebuilt = family.synchronize_private_state(rebuilt_view, reset_instance)
        rebuilt_instance = (
            rebuilt.resulting_instance(reset_instance)
            if isinstance(rebuilt, PrivateStateTransition) else reset_instance
        )
        boundary = rebuild_time + 3.0
        return {
            "broken_functional": family.derive(broken_view, reset_instance).functional,
            "channels": channels,
            "function_after_rebuild": {
                "at_boundary": family.derive(_view_at(view, boundary), rebuilt_instance).functional,
                "after_boundary": family.derive(
                    _view_at(view, math.nextafter(boundary, math.inf)), rebuilt_instance,
                ).functional,
                "before_boundary": family.derive(
                    _view_at(view, math.nextafter(boundary, -math.inf)), rebuilt_instance,
                ).functional,
            },
            "intervention": _intervention_contract(intervention),
            "rebuild": _private_contract(rebuilt),
            "reset": _private_contract(reset),
            "start": _private_contract(started),
        }
    raise ValueError(f"No production calibration evaluator for {case_id}")


def _calibration_matches(observed: object, expected: object, absolute: float, relative: float) -> bool:
    if type(observed) in (int, float) and type(expected) in (int, float):
        return math.isclose(float(observed), float(expected), abs_tol=absolute, rel_tol=relative)
    return observed == expected


def binary_entropy(p: float) -> float:
    if not 0<=p<=1: raise ValueError("Probability outside [0,1]")
    if p in (0,1): return 0.0
    return -p*math.log2(p)-(1-p)*math.log2(1-p)


def check_laws(
    samples: int = 768, *, families: tuple[LawFamily, ...] | None = None,
) -> dict[str, Any]:
    if samples<32: raise ValueError("Use at least 32 stochastic replications")
    # At a fertile empty location with only spawning, P(raw at T)=1-exp(-r T).
    c=replace(Config(),source_rate=0.3,raw_decay=0,rich_decay=0,conversion_rate=0,
              module_decay=0,regime_rate=0)
    spawned=[]
    for seed in range(900_000,900_000+samples):
        w=World(seed,c,record=False); w._die("test_fixture");w.retire()
        w.normalize_resources(np.zeros_like(w.resources))
        w.advance(2.0)
        spawned.append(int(w.resources[w.home]==1))
    p=1-math.exp(-0.3*2)
    empirical=float(np.mean(spawned));se=math.sqrt(p*(1-p)/samples)
    # For reversible RAW <-> RICH with rates gamma,delta and initial RAW:
    # P(RICH at T)=gamma/(gamma+delta)*(1-exp(-(gamma+delta)*T)).
    c=replace(Config(),source_rate=0,raw_decay=0,rich_decay=0.2,conversion_rate=0.4,
              module_decay=0,regime_rate=0)
    rich=[]
    for seed in range(910_000,910_000+samples):
        w=World(seed,c,Law((0,1)),record=False);w._die("test_fixture");w.retire()
        w.modules=[w.home,(w.home[0],w.home[1]+1),(0,0)];w._update_field()
        resources=np.zeros_like(w.resources);target=(w.home[0]-1,w.home[1]);resources[target]=1
        w.normalize_resources(resources);w.advance(3)
        rich.append(int(w.resources[target]==2))
    q=0.4/0.6*(1-math.exp(-0.6*3));estimate=float(np.mean(rich));seq=math.sqrt(q*(1-q)/samples)
    checks=[dict(name="source birth probability",analytic=p,empirical=empirical,standard_error=se,
                 z_score=(empirical-p)/se,passed=abs(empirical-p)<5*se),
            dict(name="reversible conversion probability",analytic=q,empirical=estimate,standard_error=seq,
                 z_score=(estimate-q)/seq,passed=abs(estimate-q)<5*seq)]
    selected = tuple(sorted(
        builtin_families() if families is None else families,
        key=lambda family: family.descriptor.family_id,
    ))
    ids = [family.descriptor.family_id for family in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("Calibration family selection contains a duplicate exact ID")
    family_results = []
    for family in selected:
        cases: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        family_id = family.descriptor.family_id
        required = _OFFICIAL_CALIBRATION_SUITES.get(family_id)
        declared: tuple[CalibrationCase, ...] = ()
        suite_error: dict[str, object] | None = None
        try:
            raw_cases = family.calibration_cases()
            if type(raw_cases) is not tuple:
                raise TypeError("calibration_cases() must return a tuple")
            if any(not isinstance(case, CalibrationCase) for case in raw_cases):
                raise TypeError("calibration_cases() contains a non-CalibrationCase item")
            declared = raw_cases
        except Exception as exc:
            suite_error = {
                "error_message": str(exc), "error_type": type(exc).__name__,
            }
        expected_suite = [] if required is None else [copy.deepcopy(item) for item in required]
        observed_suite = (
            suite_error if suite_error is not None
            else [_calibration_case_contract(case) for case in declared]
        )
        if required is None or suite_error is not None or observed_suite != expected_suite:
            reason = (
                "family has no benchmark-owned calibration suite"
                if required is None
                else "declared calibration suite is not the exact exhaustive benchmark suite"
            )
            failures.append({
                "case_id": "__suite__",
                "expected": expected_suite,
                "observed": copy.deepcopy(observed_suite),
                "reason": reason,
            })
            cases.append({
                "absolute_tolerance": 0.0,
                "case_id": "__suite__",
                "expected": {"suite": expected_suite},
                "kind": "benchmark-contract",
                "observed": {"suite": copy.deepcopy(observed_suite)},
                "parameters": {},
                "passed": False,
                "relative_tolerance": 0.0,
                "samples_required": 0,
            })
        else:
            declared_by_id = {case.case_id: case for case in declared}
            for required_case in required:
                case_id = str(required_case["case_id"])
                case = declared_by_id[case_id]
                expected = _expected_calibration_observation(case_id)
                try:
                    observed = _calibration_observation(family, case_id)
                except Exception as exc:
                    observed = {
                        "error_message": str(exc),
                        "error_type": type(exc).__name__,
                    }
                passed = _calibration_matches(
                    observed, expected, case.absolute_tolerance,
                    case.relative_tolerance,
                )
                case_result = {
                    "absolute_tolerance": case.absolute_tolerance,
                    "case_id": case_id,
                    "expected": expected,
                    "kind": case.kind,
                    "observed": observed,
                    "parameters": thaw_json(case.parameters),
                    "passed": passed,
                    "relative_tolerance": case.relative_tolerance,
                    "samples_required": case.samples,
                }
                cases.append(case_result)
                if not passed:
                    failures.append({
                        "case_id": case_id,
                        "expected": copy.deepcopy(expected),
                        "observed": copy.deepcopy(observed),
                        "reason": "observed mechanism contract missed benchmark-owned expectation",
                    })
        try:
            suite_fingerprint: str | None = calibration_suite_fingerprint(family)
        except Exception:
            suite_fingerprint = None
        family_results.append({
            "calibration_suite_sha256": suite_fingerprint,
            "cases": cases,
            "descriptor": family.descriptor.persistence_dict(),
            "failures": failures,
            "family_id": family_id,
            "fingerprint": fingerprint_family(family),
            "passed": not failures,
            "samples": sum(case["samples_required"] for case in cases),
        })
    legacy_passed = all(x["passed"] for x in checks)
    aggregate_passed = legacy_passed and all(row["passed"] for row in family_results)
    return {"samples_per_check":samples,"checks":checks,"families":family_results,
            "aggregate":{"family_count":len(family_results),"passed":aggregate_passed,
                         "samples":samples},"passed":aggregate_passed,
            "tolerance":"Predefined five-binomial-standard-error numerical diagnostic; not a scientific effect test.",
            "entropy_example":{"symmetric_regime_flip_probability":(1-math.exp(-2*0.02*10))/2,
                "bits_per_sample":binary_entropy((1-math.exp(-2*0.02*10))/2),
                "note":"Sampled stationary two-state chain. Shannon entropy in bits is not thermodynamic entropy."}}
