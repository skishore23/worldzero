"""Shared producer/consumer contracts for causal and transfer scoring.

These records validate interpretation boundaries; they do not authenticate a
trace. Trace replay remains responsible for verifying the underlying events.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from .laws.types import FamilyEvidence, freeze_json, thaw_json


@dataclass(frozen=True)
class CausalWitness:
    construction: int
    first_effect: int
    first_observation: int
    disruption: int
    reconstruction: int
    recurrence: int
    recurrence_observation: int
    benefit: int | None = None

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if name == "benefit" and value is None:
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"Causal witness {name} must be a nonnegative integer")
        ordered = [value for name, value in values.items() if name != "benefit"]
        if any(left >= right for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Causal witness references must follow causal order")
        if self.benefit is not None and self.benefit < self.recurrence_observation:
            raise ValueError("Causal witness benefit precedes public recurrence")

    def persistence_dict(self) -> dict[str, int]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_persistence(cls, value: Mapping[str, Any]) -> CausalWitness:
        required = set(cls.__dataclass_fields__) - {"benefit"}
        if not isinstance(value, Mapping) or set(value) not in (required, required | {"benefit"}):
            raise ValueError("Causal witness fields are invalid")
        if "benefit" in value and value["benefit"] is None:
            raise ValueError("Absent causal witness benefit must be omitted")
        return cls(**value)


@dataclass(frozen=True)
class BenchmarkEvidence:
    family: FamilyEvidence
    witness: CausalWitness | None

    def __post_init__(self) -> None:
        if not isinstance(self.family, FamilyEvidence):
            raise TypeError("Benchmark evidence requires FamilyEvidence")
        if self.witness is not None and not isinstance(self.witness, CausalWitness):
            raise TypeError("Benchmark evidence requires a CausalWitness")
        stages = thaw_json(self.family.stage_evidence)
        stages["causal_witness"] = self.witness.persistence_dict() if self.witness else None
        object.__setattr__(self, "family", replace(
            self.family, stage_evidence=stages,
            discriminating_verification=self.discriminating_verification,
            linked_benefit=self.linked_benefit,
        ))

    @property
    def discriminating_verification(self) -> bool:
        return self.witness is not None

    @property
    def linked_benefit(self) -> bool:
        return self.witness is not None and self.witness.benefit is not None

    def as_family_evidence(self) -> FamilyEvidence:
        return self.family

    def persistence_dict(self) -> dict[str, Any]:
        return self.as_family_evidence().persistence_dict()

    @classmethod
    def from_persistence(cls, value: Mapping[str, Any]) -> BenchmarkEvidence:
        family = FamilyEvidence.from_persistence(value)
        if "causal_witness" not in family.stage_evidence:
            raise ValueError("Benchmark evidence is missing its causal witness field")
        raw = family.stage_evidence["causal_witness"]
        record = cls(family, CausalWitness.from_persistence(raw) if raw is not None else None)
        if (family.discriminating_verification != record.discriminating_verification
                or family.linked_benefit != record.linked_benefit):
            raise ValueError("Benchmark evidence flags contradict its causal witness")
        return record


@dataclass(frozen=True)
class BranchOutcome:
    """Validated episode outcome with detached, lossless episode metadata."""
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise ValueError("Inheritance branch must be an episode object")
        status, survived = self.payload.get("status"), self.payload.get("survived")
        if not isinstance(status, str) or status not in {"completed", "censored", "failed"}:
            raise ValueError("Inheritance branch status is invalid")
        if ((status == "completed" and type(survived) is not bool)
                or (status != "completed" and survived is not None)):
            raise ValueError("Inheritance branch survival contradicts its status")
        object.__setattr__(self, "payload", freeze_json(self.payload))

    @property
    def completed(self) -> bool:
        return self.payload["status"] == "completed"

    @property
    def survived(self) -> bool | None:
        return self.payload.get("survived")


@dataclass(frozen=True)
class InheritanceResult:
    """Transfer scoring fields; experiment metadata stays in the enclosing row."""
    eligible: bool
    retained: BranchOutcome
    knockout: BranchOutcome
    broken: BranchOutcome

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise ValueError("Inheritance eligibility must be boolean")
        if not all(isinstance(branch, BranchOutcome) for branch in self.branches):
            raise TypeError("Inheritance requires three BranchOutcome records")

    @property
    def branches(self) -> tuple[BranchOutcome, BranchOutcome, BranchOutcome]:
        return self.retained, self.knockout, self.broken

    @property
    def completed(self) -> bool:
        return all(branch.completed for branch in self.branches)

    @property
    def status(self) -> str:
        if any(branch.payload["status"] == "failed" for branch in self.branches):
            return "failed"
        return "completed" if self.completed else "censored"

    @property
    def transfer_qualifies(self) -> bool:
        return (self.completed and self.eligible and self.retained.survived is True
                and any(branch.survived is False for branch in (self.knockout, self.broken)))

    def persistence_dict(self) -> dict[str, Any]:
        return {"status": self.status, "eligible": self.eligible,
                "results": {name: thaw_json(getattr(self, name).payload)
                            for name in ("retained", "knockout", "broken")}}

    @classmethod
    def from_outcomes(cls, results: Mapping[str, Any], *, eligible: bool) -> InheritanceResult:
        names = ("retained", "knockout", "broken")
        if not isinstance(results, Mapping) or set(results) != set(names):
            raise ValueError("Inheritance requires retained, knockout, and broken outcomes")
        return cls(eligible, *(BranchOutcome(results[name]) for name in names))

    @classmethod
    def from_persistence(cls, value: Mapping[str, Any]) -> InheritanceResult:
        if not isinstance(value, Mapping) or not {"status", "eligible", "results"} <= set(value):
            raise ValueError("Inheritance result fields are invalid")
        record = cls.from_outcomes(value["results"], eligible=value["eligible"])
        if value["status"] != record.status:
            raise ValueError("Inheritance status contradicts its branch outcomes")
        return record
