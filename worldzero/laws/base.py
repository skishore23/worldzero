"""Stable abstract interface implemented by WorldZero law families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from worldzero.kernel import Config

from .types import (
    CalibrationCase,
    ChannelSpec,
    ControlKind,
    ControlSuite,
    DerivedLawState,
    EvaluatorTrace,
    FamilyDescriptor,
    FamilyEvidence,
    FamilyInstance,
    InterventionTransition,
    JSONValue,
    KernelProposalRejection,
    LawTransition,
    ProposalDraw,
    PrivateStateTransition,
    PublicSubstrateView,
    SampleContext,
    SubstrateView,
)


class LawFamily(ABC):
    """One trusted evaluator-side hidden causal-law implementation."""

    @property
    @abstractmethod
    def descriptor(self) -> FamilyDescriptor:
        """Return the stable descriptor frozen into run identity."""

    @abstractmethod
    def sample(self, context: SampleContext) -> FamilyInstance:
        """Sample one instance exclusively from kernel-supplied named draws."""

    @abstractmethod
    def channels(self, instance: FamilyInstance, config: Config) -> tuple[ChannelSpec, ...]:
        """Declare the state-independent proposal envelope."""

    @abstractmethod
    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        """Derive evaluator-only state without mutation."""

    @abstractmethod
    def apply_proposal(
        self,
        proposal: ProposalDraw,
        view: SubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> LawTransition | None:
        """Return a closed transition for one proposal, or no transition."""

    def filter_kernel_proposal(
        self,
        proposal: ProposalDraw,
        view: SubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> KernelProposalRejection | None:
        """Reject an applicable kernel proposal through the closed filter API."""

        return None

    def synchronize_private_state(
        self,
        view: SubstrateView,
        instance: FamilyInstance,
    ) -> PrivateStateTransition | None:
        """Return a deterministic private-state update for the current substrate."""

        return None

    def internal_deadline(
        self,
        view: SubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> float | None:
        """Return the next exact simulated-time state discontinuity, if any."""

        return None

    @abstractmethod
    def project_public(
        self,
        view: PublicSubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> Mapping[str, JSONValue]:
        """Project schema-declared locally observable fields."""

    @abstractmethod
    def controls(self, instance: FamilyInstance) -> ControlSuite:
        """Declare matched null, knockout, broken, and retained controls."""

    @abstractmethod
    def intervene(
        self,
        control: ControlKind,
        view: SubstrateView,
        instance: FamilyInstance,
    ) -> InterventionTransition:
        """Describe a typed control intervention without mutating the view."""

    @abstractmethod
    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        """Extract standardized evidence without making a benchmark decision."""

    @abstractmethod
    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        """Return bounded analytic or invariant calibration cases."""


__all__ = ["LawFamily"]
