"""Built-in simulated-time delayed RAW-to-RICH transformation."""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..base import LawFamily
from ..types import (
    AccountingDelta, CalibrationCase, ChannelSpec, ControlKind, ControlSpec,
    ControlSuite, DerivedLawState, DrawRequirement, EvaluatorTrace,
    FamilyDescriptor, FamilyEvidence, FamilyInstance, InterventionTransition,
    LawTransition, ModulePositionChange, PrivateStateTransition, ProposalDraw, PublicSubstrateView,
    ResourceReplacement, SampleContext, SubstrateView, TargetDomain,
)

RAW, RICH = 1, 2
_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {}}
_MATCHING = frozenset({"material_stock", "proposal_stream", "public_substrate"})


def _pair_geometry(instance: FamilyInstance) -> tuple[tuple[int, int], str]:
    pair = instance.hidden_parameters["pair"]
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError("delayed-transformation pair is malformed")
    return (int(pair[0]), int(pair[1])), str(instance.hidden_parameters["geometry"])


def _structural(view: SubstrateView, pair: tuple[int, int], geometry: str) -> bool:
    first, second = (view.module_positions[index] for index in pair)
    if first is None or second is None:
        return False
    distance = abs(first[0] - second[0]) + abs(first[1] - second[1])
    return distance == (1 if geometry == "adjacent" else 2)


def _affected(view: SubstrateView, pair: tuple[int, int]) -> tuple[int, ...]:
    locations: set[int] = set()
    for index in pair:
        position = view.module_positions[index]
        if position is None:
            continue
        y, x = position
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < view.height and 0 <= nx < view.width:
                locations.add(ny * view.width + nx)
    return tuple(sorted(locations))


def _broken_transition(control: ControlKind, view: SubstrateView,
                       instance: FamilyInstance) -> InterventionTransition:
    pair, geometry = _pair_geometry(instance)
    if not _structural(view, pair, geometry):
        return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)
    index, other_index = pair
    old, other = view.module_positions[index], view.module_positions[other_index]
    assert old is not None and other is not None
    occupied = {position for position in view.module_positions if position is not None}
    options = [
        (y, x) for y in range(view.height) for x in range(view.width)
        if (y, x) not in occupied and abs(y - other[0]) + abs(x - other[1]) > 3
    ]
    if not options:
        raise RuntimeError("No legal ablation location")
    replacement = min(options, key=lambda position: (
        abs(position[0] - old[0]) + abs(position[1] - old[1]), position,
    ))
    return InterventionTransition(
        control, (ModulePositionChange(index, old, replacement),), AccountingDelta(),
        frozenset({"geometry_control"}), instance,
    )


class DelayedTransformationFamily(LawFamily):
    descriptor = FamilyDescriptor(
        family_id="worldzero:delayed-transformation", api_version="1.0",
        family_version="1.0.0", display_name="Delayed transformation",
        package="worldzero-research", package_version="0.3.0",
        capabilities=frozenset({"geometry_control", "private_state_transition", "resource_transition"}),
        observation_schema=_SCHEMA,
    )

    def sample(self, context: SampleContext) -> FamilyInstance:
        pair = context.named_draws.get("compatibility_pair", context.named_draws.get("legacy_pair"))
        if pair is None:
            pair = context.sample_indices("law", population_size=int(context.draw("module_count")), count=2)
        dwell = float(context.draw("dwell_duration"))
        if not math.isfinite(dwell) or dwell < 0.0:
            raise ValueError("dwell_duration must be finite and nonnegative")
        gain = float(context.named_draws.get("resource_energy_gain", 5.2))
        geometry = context.named_draws.get("compatibility_geometry", context.named_draws.get("legacy_geometry", "adjacent"))
        return FamilyInstance(
            self.descriptor.family_id, self.descriptor.family_version,
            {"dwell_duration": dwell, "geometry": geometry, "pair": pair,
             "resource_energy_gain": gain},
            {"assembled_since": None},
        )

    def channels(self, instance: FamilyInstance, config: Any) -> tuple[ChannelSpec, ...]:
        return (ChannelSpec("convert", config.conversion_rate * config.width * config.height,
                            TargetDomain.CELL,
                            (DrawRequirement.TARGET_INDEX, DrawRequirement.ACCEPTANCE_UNIFORM)),)

    def synchronize_private_state(self, view: SubstrateView,
                                  instance: FamilyInstance) -> PrivateStateTransition | None:
        pair, geometry = _pair_geometry(instance)
        structural = _structural(view, pair, geometry)
        since = instance.private_state.get("assembled_since")
        replacement = since
        if structural and since is None:
            replacement = view.simulated_time
        elif not structural and since is not None:
            replacement = None
        if replacement == since:
            return None
        return PrivateStateTransition(instance.private_state, {"assembled_since": replacement},
                                      frozenset({"private_state_transition"}))

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        pair, geometry = _pair_geometry(instance)
        structural = _structural(view, pair, geometry)
        since = instance.private_state.get("assembled_since")
        mature = (structural and isinstance(since, (int, float)) and
                  view.simulated_time >= float(since) + float(instance.hidden_parameters["dwell_duration"]))
        functional = instance.enabled and mature
        return DerivedLawState({"structural": structural, "mature": mature}, functional,
                               _affected(view, pair) if functional else ())

    def internal_deadline(
        self, view: SubstrateView, instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> float | None:
        since = instance.private_state.get("assembled_since")
        if (
            not instance.enabled
            or derived.state.get("structural") is not True
            or derived.state.get("mature") is True
            or not isinstance(since, (int, float))
        ):
            return None
        boundary = float(since) + float(instance.hidden_parameters["dwell_duration"])
        return boundary if boundary > view.simulated_time else None

    def apply_proposal(self, proposal: ProposalDraw, view: SubstrateView,
                       instance: FamilyInstance, derived: DerivedLawState) -> LawTransition | None:
        if proposal.channel_id != "convert" or not derived.functional:
            return None
        y, x = divmod(proposal.target_index, view.width)
        if y >= view.height or view.resources[y][x] != RAW or proposal.target_index not in derived.affected_locations:
            return None
        return LawTransition(
            (ResourceReplacement(proposal.target_index, RAW, RICH),),
            AccountingDelta(0, float(instance.hidden_parameters["resource_energy_gain"])),
            frozenset({"resource_transition"}),
        )

    def project_public(self, view: PublicSubstrateView, instance: FamilyInstance,
                       derived: DerivedLawState) -> Mapping[str, object]:
        return {}

    def controls(self, instance: FamilyInstance) -> ControlSuite:
        return ControlSuite(*(ControlSpec(kind, _MATCHING) for kind in ControlKind))

    def intervene(self, control: ControlKind, view: SubstrateView,
                  instance: FamilyInstance) -> InterventionTransition:
        control = ControlKind(control)
        if control in {ControlKind.NULL, ControlKind.KNOCKOUT}:
            disabled = FamilyInstance(instance.family_id, instance.family_version,
                                      instance.hidden_parameters, instance.private_state, False)
            return InterventionTransition(control, (), AccountingDelta(), frozenset(), disabled)
        if control is ControlKind.BROKEN:
            return _broken_transition(control, view, instance)
        return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        assemblies = tuple(i for i, event in enumerate(trace.events) if event.get("kind") == "assembly")
        conversions = tuple(i for i, event in enumerate(trace.events)
                            if event.get("kind") == "physics" and event.get("event") == "convert")
        functional = trace.terminal.get("functional") is True
        rich_symbol = trace.terminal.get("rich_symbol")
        width = trace.terminal.get("width")
        consequence_refs: list[int] = []
        benefit_refs: list[int] = []
        for effect_index in conversions:
            target = trace.events[effect_index].get("target")
            if type(target) is not int or type(width) is not int or width <= 0:
                continue
            position = [target // width, target % width]
            for index in range(effect_index + 1, len(trace.events)):
                event = trace.events[index]
                if event.get("kind") not in {
                    "policy_observation", "policy_result", "policy_record",
                }:
                    continue
                observation = event.get("observation")
                result = event.get("result")
                visible = event.get("kind") in {
                    "policy_observation", "policy_record",
                } and isinstance(observation, Mapping) and any(
                    tuple(cell.get("position", ())) == tuple(position)
                    and any(
                        isinstance(item, Mapping) and item.get("id") == rich_symbol
                        for item in cell.get("objects", ())
                    )
                    for cell in observation.get("local", ())
                    if isinstance(cell, Mapping)
                )
                consumed = event.get("kind") in {"policy_result", "policy_record"} and (
                    isinstance(result, Mapping)
                    and result.get("status") == "consumed"
                    and result.get("object_id") == rich_symbol
                    and isinstance(result.get("gross_energy"), (int, float))
                    and float(result["gross_energy"]) > 0.0
                )
                if visible or consumed:
                    consequence_refs.append(index)
                if consumed:
                    benefit_refs.append(index)
        return FamilyEvidence(
            {"assemblies": len(assemblies), "mature": functional,
             "conversions": len(conversions)},
            tuple(sorted(set((*assemblies, *conversions, *consequence_refs)))),
            "model_placement" if assemblies else ("pre_existing" if functional else "none"),
            bool(assemblies), functional or bool(conversions), bool(conversions),
            bool(consequence_refs), bool(
                assemblies and conversions and min(assemblies) < min(conversions)
                and consequence_refs and min(conversions) < min(consequence_refs)
            ), False, functional, bool(benefit_refs),
            {"delayed_effect": bool(conversions)},
        )

    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return (
            CalibrationCase("exact-dwell-boundary", "analytic", True,
                            parameters={"clock": "simulated_time"}),
            CalibrationCase("break-resets-maturation", "invariant", True),
        )


__all__ = ["DelayedTransformationFamily"]
