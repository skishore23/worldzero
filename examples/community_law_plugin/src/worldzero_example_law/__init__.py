"""Small, installable WorldZero preservation-law example.

The plugin sees only public immutable SDK records.  It never imports or
receives the mutable kernel ``World``.
"""

from __future__ import annotations

from typing import Any, Mapping

from worldzero.laws import (
    AccountingDelta,
    CalibrationCase,
    ChannelSpec,
    ControlKind,
    ControlSpec,
    ControlSuite,
    DerivedLawState,
    EvaluatorTrace,
    EventEvidence,
    FamilyDescriptor,
    FamilyEvidence,
    FamilyInstance,
    InterventionTransition,
    KernelProposalRejection,
    LawFamily,
    LawTransition,
    ModulePositionChange,
    ProposalDraw,
    PublicSubstrateView,
    ResourcePreservation,
    SampleContext,
    SubstrateView,
)


_MATCHING = frozenset({"material_stock", "proposal_stream", "public_substrate"})


def _pair(instance: FamilyInstance) -> tuple[int, int]:
    pair = instance.hidden_parameters["pair"]
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError("sampled pair is malformed")
    return int(pair[0]), int(pair[1])


def _structural(view: SubstrateView, instance: FamilyInstance) -> bool:
    first, second = (view.module_positions[index] for index in _pair(instance))
    return (
        first is not None
        and second is not None
        and abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
    )


def _affected(view: SubstrateView, instance: FamilyInstance) -> tuple[int, ...]:
    if not _structural(view, instance):
        return ()
    cells: set[int] = set()
    for module_index in _pair(instance):
        position = view.module_positions[module_index]
        assert position is not None
        for row in range(max(0, position[0] - 1), min(view.height, position[0] + 2)):
            for column in range(max(0, position[1] - 1), min(view.width, position[1] + 2)):
                if abs(row - position[0]) + abs(column - position[1]) <= 1:
                    cells.add(row * view.width + column)
    return tuple(sorted(cells))


class PreserverFamily(LawFamily):
    """An adjacent hidden module pair preserves nearby RAW resources."""

    descriptor = FamilyDescriptor(
        family_id="example_org:preserver",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Example RAW preserver",
        package="worldzero-example-law",
        package_version="0.1.0",
        capabilities=frozenset({
            "event_evidence",
            "geometry_control",
            "proposal_filter",
            "resource_preservation",
        }),
        observation_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
        documentation_url=None,
    )

    def sample(self, context: SampleContext) -> FamilyInstance:
        return FamilyInstance(
            self.descriptor.family_id,
            self.descriptor.family_version,
            {
                "geometry": "adjacent",
                "pair": context.sample_indices(
                    "law", population_size=int(context.draw("module_count")), count=2
                ),
            },
            {},
        )

    def channels(self, instance: FamilyInstance, config: Any) -> tuple[ChannelSpec, ...]:
        return ()

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        locations = _affected(view, instance)
        structural = bool(locations)
        return DerivedLawState(
            {"structural": structural},
            structural and instance.enabled,
            locations,
        )

    def apply_proposal(
        self,
        proposal: ProposalDraw,
        view: SubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> LawTransition | None:
        return None

    def filter_kernel_proposal(
        self,
        proposal: ProposalDraw,
        view: SubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> KernelProposalRejection | None:
        if (
            proposal.channel_id != "raw_decay"
            or not instance.enabled
            or not derived.functional
            or proposal.target_index not in derived.affected_locations
        ):
            return None
        return KernelProposalRejection(
            proposal,
            (
                ResourcePreservation(proposal.target_index, 1),
                EventEvidence(
                    "preserved_raw",
                    proposal.target_index,
                    {"accounting": "zero"},
                ),
            ),
            frozenset({"proposal_filter", "resource_preservation", "event_evidence"}),
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
            disabled = FamilyInstance(
                instance.family_id,
                instance.family_version,
                instance.hidden_parameters,
                instance.private_state,
                enabled=False,
            )
            return InterventionTransition(
                control, (), AccountingDelta(), frozenset(), disabled
            )
        if control is ControlKind.BROKEN and _structural(view, instance):
            module_index = _pair(instance)[0]
            old = view.module_positions[module_index]
            other = view.module_positions[_pair(instance)[1]]
            assert old is not None and other is not None
            occupied = {position for position in view.module_positions if position is not None}
            candidates = [
                (row, column)
                for row in range(view.height)
                for column in range(view.width)
                if (row, column) not in occupied
                and abs(row - other[0]) + abs(column - other[1]) > 1
            ]
            replacement = min(
                candidates,
                key=lambda position: (
                    abs(position[0] - old[0]) + abs(position[1] - old[1]),
                    position,
                ),
            )
            return InterventionTransition(
                control,
                (ModulePositionChange(module_index, old, replacement),),
                AccountingDelta(),
                frozenset({"geometry_control"}),
                instance,
            )
        return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        references = tuple(
            index
            for index, event in enumerate(trace.events)
            if event.get("kind") == "family_evidence"
            and event.get("event") == "preserved_raw"
        )
        return FamilyEvidence(
            {"preserved_raw_events": len(references)},
            event_references=references,
            function_observed=bool(references),
            effect_observed=bool(references),
            diagnostics={"preservation_events": len(references)},
        )

    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return (
            CalibrationCase(
                "preservation-zero-accounting",
                "validator_contract",
                True,
                parameters={"contract": "transition_accounting"},
            ),
        )


def family() -> LawFamily:
    """Entry-point factory for ``example_org:preserver``."""

    return PreserverFamily()


__all__ = ["PreserverFamily", "family"]
