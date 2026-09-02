"""Third-round adversarial regressions for the public family validator."""

from __future__ import annotations

import sys

from worldzero.laws import (
    AccountingDelta,
    ControlKind,
    DerivedLawState,
    EvaluatorTrace,
    FamilyEvidence,
    FamilyInstance,
    InterventionTransition,
    ModulePositionChange,
    SampleContext,
    SubstrateView,
    thaw_json,
)

from test_task6_fix_round2 import (
    ConformingCommunityFamily,
    _assert_rejected,
    _validate,
)


class _DetachedTracePayload:
    __slots__ = ("events", "terminal", "link")

    def __init__(self) -> None:
        self.events: object = None
        self.terminal: object = None
        self.link: object = self


class _NestedStorage:
    __slots__ = ("inner", "link")

    def __init__(self) -> None:
        self.inner = _DetachedTracePayload()
        self.link: object = self


class DeepCustomDetachedTraceFamily(ConformingCommunityFamily):
    retained = _NestedStorage()

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        type(self).retained.inner.events = [
            thaw_json(event) for event in trace.events
        ]
        type(self).retained.inner.terminal = thaw_json(trace.terminal)
        return super().evaluate(trace)


def test_sensitive_detached_subgraph_inside_nested_slotted_cycle_is_rejected() -> None:
    _assert_rejected(DeepCustomDetachedTraceFamily(), "callback_isolation")


class MultiOrientationBrokenFamily(ConformingCommunityFamily):
    @staticmethod
    def _orientation(view: SubstrateView) -> str:
        occupied = set(view.module_positions)
        if occupied == {(0, 0), (0, 1), (0, 2)}:
            return "horizontal"
        if occupied == {(0, 0), (1, 0), (2, 0)}:
            return "vertical"
        return "none"

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        orientation = self._orientation(view)
        return DerivedLawState(
            {"structural": orientation != "none", "orientation": orientation},
            False,
        )

    def intervene(
        self,
        control: ControlKind,
        view: SubstrateView,
        instance: FamilyInstance,
    ) -> InterventionTransition:
        if control is ControlKind.BROKEN:
            orientation = self._orientation(view)
            if orientation == "horizontal":
                old = view.module_positions[0]
                return InterventionTransition(
                    control,
                    (ModulePositionChange(0, old, (8, 12)),),
                    AccountingDelta(),
                    frozenset({"geometry_control"}),
                    instance,
                )
            if orientation == "vertical":
                old = view.module_positions[0]
                return InterventionTransition(
                    control,
                    (ModulePositionChange(0, old, (999, 999)),),
                    AccountingDelta(),
                    frozenset({"geometry_control"}),
                    instance,
                )
            return InterventionTransition(
                control, (), AccountingDelta(), frozenset(), instance,
            )
        return super().intervene(control, view, instance)


def test_every_applicable_structural_orientation_is_exercised() -> None:
    _assert_rejected(MultiOrientationBrokenFamily(), "controls")


class MultiOrientationRetainedFamily(MultiOrientationBrokenFamily):
    def intervene(
        self,
        control: ControlKind,
        view: SubstrateView,
        instance: FamilyInstance,
    ) -> InterventionTransition:
        if control is ControlKind.BROKEN:
            orientation = self._orientation(view)
            if orientation != "none":
                old = view.module_positions[0]
                return InterventionTransition(
                    control,
                    (ModulePositionChange(0, old, (8, 12)),),
                    AccountingDelta(),
                    frozenset({"geometry_control"}),
                    instance,
                )
            return InterventionTransition(
                control, (), AccountingDelta(), frozenset(), instance,
            )
        if control is ControlKind.RETAINED and self._orientation(view) == "vertical":
            old = view.module_positions[0]
            return InterventionTransition(
                control,
                (ModulePositionChange(0, old, (8, 12)),),
                AccountingDelta(),
                frozenset({"geometry_control"}),
                instance,
            )
        return super().intervene(control, view, instance)


def test_every_control_is_exercised_for_each_structural_orientation() -> None:
    _assert_rejected(MultiOrientationRetainedFamily(), "controls")


_PROPERTY_READS = 0


class _SlotBase:
    __slots__ = ("payload",)

    def __init__(self) -> None:
        descriptor = vars(_SlotBase)["payload"]
        descriptor.__set__(self, {"ordinary": True})


class _SlotShadow(_SlotBase):
    @property
    def payload(self) -> object:
        global _PROPERTY_READS
        _PROPERTY_READS += 1
        raise AssertionError("slot-shadowing property executed")


class SlotShadowConformingFamily(ConformingCommunityFamily):
    retained = _SlotShadow()


def test_slot_shadowing_property_is_never_invoked() -> None:
    global _PROPERTY_READS
    _PROPERTY_READS = 0
    report = _validate(SlotShadowConformingFamily())
    assert report["passed"] is True, report
    assert _PROPERTY_READS == 0


def test_unrelated_preexisting_module_callback_record_is_ignored() -> None:
    module = sys.modules[ConformingCommunityFamily.__module__]
    setattr(
        module,
        "_worldzero_unrelated_context",
        SampleContext({"module_count": 3}, {"law": 7}),
    )
    try:
        report = _validate(ConformingCommunityFamily())
    finally:
        delattr(module, "_worldzero_unrelated_context")
    assert report["passed"] is True, report


class ModuleGlobalRetentionFamily(ConformingCommunityFamily):
    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        module = sys.modules[type(self).__module__]
        setattr(module, "_worldzero_callback_record", trace)
        return super().evaluate(trace)


def test_callback_stored_module_global_is_rejected() -> None:
    module = sys.modules[ModuleGlobalRetentionFamily.__module__]
    try:
        _assert_rejected(ModuleGlobalRetentionFamily(), "callback_isolation")
    finally:
        if hasattr(module, "_worldzero_callback_record"):
            delattr(module, "_worldzero_callback_record")
