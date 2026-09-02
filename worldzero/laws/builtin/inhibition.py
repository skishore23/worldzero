"""Built-in local inhibition of kernel RAW-decay proposals."""

from __future__ import annotations

from typing import Any, Mapping

from ..base import LawFamily
from ..types import (
    AccountingDelta, CalibrationCase, ControlKind, ControlSpec, ControlSuite,
    DerivedLawState, EventEvidence, EvaluatorTrace, FamilyDescriptor,
    FamilyEvidence, FamilyInstance, InterventionTransition,
    KernelProposalRejection, LawTransition, ModulePositionChange, ProposalDraw,
    PublicSubstrateView, ResourcePreservation, SampleContext, SubstrateView,
)

RAW = 1
_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {}}
_MATCHING = frozenset({"material_stock", "proposal_stream", "public_substrate"})


def _pair_geometry(instance: FamilyInstance) -> tuple[tuple[int, int], str]:
    pair = instance.hidden_parameters["pair"]
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError("inhibition pair is malformed")
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


def _broken_transition(control: ControlKind, view: SubstrateView, instance: FamilyInstance) -> InterventionTransition:
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


class InhibitionFamily(LawFamily):
    descriptor = FamilyDescriptor(
        family_id="worldzero:inhibition", api_version="1.0", family_version="1.0.0",
        display_name="Inhibition", package="worldzero-research", package_version="0.3.0",
        capabilities=frozenset({
            "event_evidence", "geometry_control", "proposal_filter", "resource_preservation",
        }),
        observation_schema=_SCHEMA,
    )

    def sample(self, context: SampleContext) -> FamilyInstance:
        pair = context.named_draws.get("compatibility_pair", context.named_draws.get("legacy_pair"))
        if pair is None:
            pair = context.sample_indices("law", population_size=int(context.draw("module_count")), count=2)
        geometry = context.named_draws.get("compatibility_geometry", context.named_draws.get("legacy_geometry", "adjacent"))
        return FamilyInstance(self.descriptor.family_id, self.descriptor.family_version,
                              {"geometry": geometry, "pair": pair}, {})

    def channels(self, instance: FamilyInstance, config: Any) -> tuple[()]:
        return ()

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        pair, geometry = _pair_geometry(instance)
        structural = _structural(view, pair, geometry)
        functional = instance.enabled and structural
        return DerivedLawState({"structural": structural}, functional,
                               _affected(view, pair) if functional else ())

    def apply_proposal(self, proposal: ProposalDraw, view: SubstrateView, instance: FamilyInstance,
                       derived: DerivedLawState) -> LawTransition | None:
        return None

    def filter_kernel_proposal(self, proposal: ProposalDraw, view: SubstrateView,
                               instance: FamilyInstance, derived: DerivedLawState) -> KernelProposalRejection | None:
        if proposal.channel_id != "raw_decay" or not derived.functional:
            return None
        y, x = divmod(proposal.target_index, view.width)
        if y >= view.height or view.resources[y][x] != RAW or proposal.target_index not in derived.affected_locations:
            return None
        return KernelProposalRejection(
            proposal,
            (ResourcePreservation(proposal.target_index, RAW),
             EventEvidence("inhibited_proposal", proposal.target_index,
                           {"proposal_index": proposal.proposal_index})),
            frozenset({"proposal_filter", "resource_preservation", "event_evidence"}),
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
        inhibited = tuple(i for i, event in enumerate(trace.events)
                          if event.get("kind") == "family_evidence" and event.get("event") == "inhibited_proposal")
        functional = trace.terminal.get("functional") is True
        raw_symbol = trace.terminal.get("raw_symbol")
        width = trace.terminal.get("width")
        consequence_refs: list[int] = []
        benefit_refs: list[int] = []
        for effect_index in inhibited:
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
                        isinstance(item, Mapping) and item.get("id") == raw_symbol
                        for item in cell.get("objects", ())
                    )
                    for cell in observation.get("local", ())
                    if isinstance(cell, Mapping)
                )
                consumed = event.get("kind") in {"policy_result", "policy_record"} and (
                    isinstance(result, Mapping)
                    and result.get("status") == "consumed"
                    and result.get("object_id") == raw_symbol
                    and isinstance(result.get("gross_energy"), (int, float))
                    and float(result["gross_energy"]) > 0.0
                )
                if visible or consumed:
                    consequence_refs.append(index)
                if consumed:
                    benefit_refs.append(index)
        return FamilyEvidence(
            {"assemblies": len(assemblies), "inhibited_proposals": len(inhibited)},
            tuple(sorted(set((*assemblies, *inhibited, *consequence_refs)))),
            "model_placement" if assemblies else ("pre_existing" if functional else "none"),
            bool(assemblies), functional or bool(inhibited), bool(inhibited),
            bool(consequence_refs), bool(
                assemblies and inhibited and min(assemblies) < min(inhibited)
                and consequence_refs and min(inhibited) < min(consequence_refs)
            ), False, functional, bool(benefit_refs),
            {"preservation_effect": bool(inhibited)},
        )

    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return (
            CalibrationCase("inside-field-raw-decay-rejection", "invariant", True,
                            parameters={"accounting_delta": [0, 0.0]}),
            CalibrationCase("outside-field-matched-decay", "invariant", True),
        )


__all__ = ["InhibitionFamily"]
