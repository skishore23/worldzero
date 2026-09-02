"""Behavior-compatible matched-null law family."""

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
    SampleContext,
    SubstrateView,
    TargetDomain,
)


_EMPTY_OBSERVATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}
_MATCHING = frozenset({"material_stock", "proposal_stream", "public_substrate"})


def _pair_geometry(instance: FamilyInstance) -> tuple[tuple[int, int], str]:
    pair_value = instance.hidden_parameters["pair"]
    if not isinstance(pair_value, tuple) or len(pair_value) != 2:
        raise ValueError("null pair is malformed")
    return (int(pair_value[0]), int(pair_value[1])), str(instance.hidden_parameters["geometry"])


def _structural(view: SubstrateView, pair: tuple[int, int], geometry: str) -> bool:
    first, second = (view.module_positions[index] for index in pair)
    if first is None or second is None:
        return False
    distance = abs(first[0] - second[0]) + abs(first[1] - second[1])
    return distance == (1 if geometry == "adjacent" else 2)


class NullFamily(LawFamily):
    """Matched nuisance/proposal process with no mechanism effect."""

    descriptor = FamilyDescriptor(
        family_id="worldzero:null",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Matched null",
        package="worldzero-research",
        package_version="0.3.0",
        capabilities=frozenset({"geometry_control"}),
        observation_schema=_EMPTY_OBSERVATION_SCHEMA,
        documentation_url=None,
    )

    def sample(self, context: SampleContext) -> FamilyInstance:
        pair = context.named_draws.get("compatibility_pair", context.named_draws.get("legacy_pair"))
        if pair is None:
            pair = context.sample_indices(
                "law", population_size=int(context.draw("module_count")), count=2,
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
                "geometry": context.named_draws.get(
                    "compatibility_geometry", context.named_draws.get("legacy_geometry", "adjacent")
                ),
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
        return DerivedLawState({"structural": _structural(view, pair, geometry)}, False, ())

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
            return InterventionTransition(
                control,
                (ModulePositionChange(index, old, replacement),),
                AccountingDelta(),
                frozenset({"geometry_control"}),
                instance,
            )
        return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        return FamilyEvidence(
            {"conversions": 0},
            diagnostics={"mechanism_effect": False},
        )

    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return (
            CalibrationCase(
                "matched-conversion-envelope",
                "invariant",
                True,
                samples=1,
                parameters={"channel": "convert", "mechanism_effect": False},
            ),
        )


__all__ = ["NullFamily"]
