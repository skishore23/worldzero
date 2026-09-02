"""Behavior-compatible built-in catalysis law family."""

from __future__ import annotations

from typing import Any, Mapping

from ..base import LawFamily
from ..types import (
    AccountingDelta,
    CalibrationCase,
    ChannelSpec,
    ControlKind,
    ControlSpec,
    ControlSuite,
    DerivedLawState,
    DrawRequirement,
    EvaluatorTrace,
    FamilyDescriptor,
    FamilyEvidence,
    FamilyInstance,
    InterventionTransition,
    LawTransition,
    ModulePositionChange,
    ProposalDraw,
    PublicSubstrateView,
    ResourceReplacement,
    SampleContext,
    SubstrateView,
    TargetDomain,
)


EMPTY, RAW, RICH = 0, 1, 2
_EMPTY_OBSERVATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}
_MATCHING = frozenset({"material_stock", "proposal_stream", "public_substrate"})


def _pair_geometry(instance: FamilyInstance) -> tuple[tuple[int, int], str]:
    pair_value = instance.hidden_parameters["pair"]
    if not isinstance(pair_value, tuple) or len(pair_value) != 2:
        raise ValueError("catalysis pair is malformed")
    pair = (int(pair_value[0]), int(pair_value[1]))
    geometry = str(instance.hidden_parameters["geometry"])
    return pair, geometry


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


class CatalysisFamily(LawFamily):
    """Legacy RAW-to-RICH conversion under a hidden pair geometry."""

    descriptor = FamilyDescriptor(
        family_id="worldzero:catalysis",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Catalysis",
        package="worldzero-research",
        package_version="0.3.0",
        capabilities=frozenset({"geometry_control", "resource_transition"}),
        observation_schema=_EMPTY_OBSERVATION_SCHEMA,
        documentation_url=None,
    )

    def sample(self, context: SampleContext) -> FamilyInstance:
        pair = context.named_draws.get("compatibility_pair", context.named_draws.get("legacy_pair"))
        if pair is None:
            pair = context.sample_indices(
                "law", population_size=int(context.draw("module_count")), count=2,
            )
        geometry = context.named_draws.get(
            "compatibility_geometry", context.named_draws.get("legacy_geometry", "adjacent")
        )
        energy_gain = context.named_draws.get("resource_energy_gain")
        if energy_gain is None:
            energy_gain = (
                float(context.draw("rich_energy")) - float(context.draw("raw_energy"))
                if "rich_energy" in context.named_draws and "raw_energy" in context.named_draws
                else 5.2
            )
        return FamilyInstance(
            self.descriptor.family_id,
            self.descriptor.family_version,
            {
                "geometry": geometry,
                "pair": pair,
                "resource_energy_gain": energy_gain,
            },
            {},
        )

    def channels(self, instance: FamilyInstance, config: Any) -> tuple[ChannelSpec, ...]:
        return (
            ChannelSpec(
                "convert",
                config.conversion_rate * config.width * config.height,
                TargetDomain.CELL,
                (DrawRequirement.TARGET_INDEX, DrawRequirement.ACCEPTANCE_UNIFORM),
            ),
        )

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        pair, geometry = _pair_geometry(instance)
        structural = _structural(view, pair, geometry)
        functional = instance.enabled and structural
        return DerivedLawState(
            {"structural": structural},
            functional,
            _affected(view, pair) if functional else (),
        )

    def apply_proposal(
        self,
        proposal: ProposalDraw,
        view: SubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> LawTransition | None:
        if proposal.channel_id != "convert" or not derived.functional:
            return None
        y, x = divmod(proposal.target_index, view.width)
        if y >= view.height or view.resources[y][x] != RAW or proposal.target_index not in derived.affected_locations:
            return None
        return LawTransition(
            (ResourceReplacement(proposal.target_index, RAW, RICH),),
            AccountingDelta(
                material_delta=0,
                energy_delta=float(instance.hidden_parameters["resource_energy_gain"]),
            ),
            frozenset({"resource_transition"}),
        )

    def project_public(
        self,
        view: PublicSubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> Mapping[str, object]:
        return {}

    def controls(self, instance: FamilyInstance) -> ControlSuite:
        return ControlSuite(
            null=ControlSpec(ControlKind.NULL, _MATCHING),
            knockout=ControlSpec(ControlKind.KNOCKOUT, _MATCHING),
            broken=ControlSpec(ControlKind.BROKEN, _MATCHING),
            retained=ControlSpec(ControlKind.RETAINED, _MATCHING),
        )

    def intervene(
        self,
        control: ControlKind,
        view: SubstrateView,
        instance: FamilyInstance,
    ) -> InterventionTransition:
        control = ControlKind(control)
        if control in {ControlKind.NULL, ControlKind.KNOCKOUT}:
            result = FamilyInstance(
                instance.family_id,
                instance.family_version,
                instance.hidden_parameters,
                instance.private_state,
                enabled=False,
            )
            return InterventionTransition(control, (), AccountingDelta(), frozenset(), result)
        if control is ControlKind.BROKEN:
            pair, geometry = _pair_geometry(instance)
            if not _structural(view, pair, geometry):
                return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)
            index = pair[0]
            old = view.module_positions[index]
            other = view.module_positions[pair[1]]
            if old is None or other is None:  # pragma: no cover - structural proves this
                return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)
            occupied = {position for position in view.module_positions if position is not None}
            options = [
                (y, x)
                for y in range(view.height)
                for x in range(view.width)
                if (y, x) not in occupied and abs(y - other[0]) + abs(x - other[1]) > 3
            ]
            if not options:
                raise RuntimeError("No legal ablation location")
            replacement = min(
                options,
                key=lambda position: (
                    abs(position[0] - old[0]) + abs(position[1] - old[1]),
                    position,
                ),
            )
            operation = ModulePositionChange(index, old, replacement)
            return InterventionTransition(
                control,
                (operation,),
                AccountingDelta(),
                frozenset({"geometry_control"}),
                instance,
            )
        return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        assembly_refs = tuple(
            index for index, event in enumerate(trace.events) if event.get("kind") == "assembly"
        )
        conversion_refs = tuple(
            index
            for index, event in enumerate(trace.events)
            if event.get("kind") == "physics" and event.get("event") == "convert"
        )
        references = tuple(sorted((*assembly_refs, *conversion_refs)))
        terminal_function = trace.terminal.get("functional") is True
        assembly_times = [trace.events[index].get("time") for index in assembly_refs]
        conversion_times = [trace.events[index].get("time") for index in conversion_refs]
        model_drop = any(
            event.get("kind") == "action"
            and event.get("status") == "dropped"
            and isinstance(event.get("action"), Mapping)
            and event["action"].get("type") == "DROP"
            and event.get("time") in assembly_times
            for event in trace.events
        )
        if assembly_refs:
            origin = "model_drop" if model_drop else "model_placement"
        elif terminal_function and any(event.get("kind") == "death_drop" for event in trace.events):
            origin = "death_drop"
        elif terminal_function:
            origin = "pre_existing"
        else:
            origin = "none"
        rich_symbol = trace.terminal.get("rich_symbol")
        consequence_refs = tuple(
            index for index, event in enumerate(trace.events)
            if event.get("kind") in {"policy_observation", "policy_record"}
            and any(
                item.get("id") == rich_symbol
                for cell in event.get("observation", {}).get("local", ())
                if isinstance(cell, Mapping)
                for item in cell.get("objects", ())
                if isinstance(item, Mapping)
            )
        )
        consequence = bool(consequence_refs)
        action_kinds = [
            event.get("action", {}).get("type")
            for event in trace.events
            if event.get("kind") == "action" and isinstance(event.get("action"), Mapping)
        ]
        discriminating = len(assembly_refs) >= 2 and "PICK" in action_kinds
        linked_refs = tuple(
            index for index, event in enumerate(trace.events)
            if event.get("kind") in {"policy_result", "policy_record"}
            and isinstance(event.get("result"), Mapping)
            and event["result"].get("status") == "consumed"
            and event["result"].get("object_id") == rich_symbol
            and conversion_refs and index > min(conversion_refs)
        )
        linked = bool(linked_refs)
        preceded = bool(
            assembly_refs and conversion_refs and consequence_refs
            and min(assembly_refs) < min(conversion_refs) < min(consequence_refs)
        )
        return FamilyEvidence(
            {
                "assemblies": len(assembly_refs),
                "conversions": len(conversion_refs),
                "terminal_function": terminal_function,
            },
            event_references=tuple(sorted((*references, *consequence_refs))),
            origin=origin,
            structure_constructed=bool(assembly_refs),
            function_observed=bool(conversion_refs) or terminal_function,
            effect_observed=bool(conversion_refs),
            relevant_consequence_observed=consequence,
            intervention_preceded_consequence=preceded,
            discriminating_verification=discriminating,
            retained_or_reconstructed=terminal_function,
            linked_benefit=linked,
            diagnostics={"conversion_events": len(conversion_refs)},
        )

    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return (
            CalibrationCase(
                "legacy-conversion-rate",
                "analytic",
                0.45,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                samples=1,
                parameters={"channel": "convert"},
            ),
            CalibrationCase(
                "conversion-accounting",
                "invariant",
                {"energy_delta": 5.2, "material_delta": 0},
                samples=1,
            ),
        )


__all__ = ["CatalysisFamily"]
