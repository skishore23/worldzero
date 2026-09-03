"""Strategy-neutral participant agent contract for WorldZero."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import importlib
import json
import math
import re
from typing import Any, Protocol

from .evidence_ledger import EVIDENCE_LEDGER_SCHEMA, canonical_ledger


class WorldZeroAgent(Protocol):
    """Minimal lifecycle implemented by participant-owned agents."""

    def reset(self, context: dict[str, Any]) -> None: ...

    def act(self, observation: dict[str, Any]) -> dict[str, Any]: ...

    def observe_result(self, result: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class AgentContractError(ValueError):
    """Raised when participant code does not implement the public contract."""


AgentFactory = Callable[[], WorldZeroAgent]

_FACTORY_REFERENCE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r":[A-Za-z_][A-Za-z0-9_]*$"
)
_FINDING_STATUSES = frozenset({
    "supported", "no_mechanism", "insufficient_evidence",
})


def _detached_json(value: Any, *, path: str) -> Any:
    """Return a detached finite-JSON copy or raise a public contract error."""

    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
        detached = json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AgentContractError(f"{path} must be finite JSON") from exc
    return detached


def _number(value: Any, *, path: str, integer: bool = False) -> int | float:
    valid = type(value) is int if integer else (
        type(value) in (int, float) and math.isfinite(value)
    )
    if not valid or value < 0:
        kind = "nonnegative integer" if integer else "finite nonnegative number"
        raise AgentContractError(f"{path} must be a {kind}")
    return value


def agent_context(
    *,
    suite: str,
    scoring_profile: str,
    episode_id: str,
    agent_seed: int,
    split: str,
    max_decisions: int,
    lifespan: float,
) -> dict[str, Any]:
    """Build the closed public context supplied at the start of one episode."""

    for name, value in (
        ("suite", suite),
        ("scoring_profile", scoring_profile),
        ("episode_id", episode_id),
    ):
        if not isinstance(value, str) or not value:
            raise AgentContractError(f"{name} must be a nonempty string")
    if split not in {"dev", "test"}:
        raise AgentContractError("split must be dev or test")
    _number(agent_seed, path="agent_seed", integer=True)
    _number(max_decisions, path="max_decisions", integer=True)
    _number(lifespan, path="lifespan")
    return {
        "schema": "worldzero-agent-context-v1",
        "suite": suite,
        "scoring_profile": scoring_profile,
        "episode_id": episode_id,
        "agent_seed": agent_seed,
        "action_schema": {
            "type": "object",
            "required": ["type"],
            "description": "One currently available primitive action from the observation.",
        },
        "finding_schema": {
            "type": "object",
            "required": ["status"],
            "additionalProperties": False,
            "properties": {
                "status": {"enum": sorted(_FINDING_STATUSES)},
            },
        },
        "budgets": {
            "max_decisions": max_decisions,
            "lifespan": lifespan,
        },
        "split": split,
    }


def _valid_finding(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"status"}
        and value["status"] in _FINDING_STATUSES
    )


def _valid_ledger(value: Any) -> bool:
    try:
        canonical_ledger(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


class AgentPolicyAdapter:
    """Adapt a participant agent to the kernel's existing policy interface."""

    def __init__(
        self, factory: AgentFactory, context: Mapping[str, Any], *, name: str,
    ) -> None:
        if not callable(factory):
            raise AgentContractError("agent factory must be callable")
        try:
            agent = factory()
        except Exception as exc:
            raise AgentContractError("agent factory raised during construction") from exc
        methods = ("reset", "act", "observe_result", "close")
        if any(not callable(getattr(agent, method, None)) for method in methods):
            raise AgentContractError(
                "agent factory must return an object implementing reset, act, "
                "observe_result, and close"
            )
        if not isinstance(name, str) or not name:
            raise AgentContractError("agent name must be a nonempty string")
        self.name = name
        self._agent = agent
        self._closed = False
        self.finding = {"status": "insufficient_evidence"}
        self.contract_errors = 0
        detached_context = _detached_json(dict(context), path="agent context")
        try:
            self._agent.reset(detached_context)
        except Exception:
            self._closed = True
            self._agent.close()
            raise

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        detached = _detached_json(observation, path="observation")
        decision = self._agent.act(detached)
        try:
            decision = _detached_json(decision, path="agent decision")
        except AgentContractError:
            self.contract_errors += 1
            return {"action": {"type": "WAIT"}, "memory": "", "invalid": True}
        if (
            not isinstance(decision, dict)
            or "action" not in decision
            or not set(decision) <= {"action", "finding", "ledger"}
            or not isinstance(decision["action"], dict)
            or not isinstance(decision["action"].get("type"), str)
            or (
                "finding" in decision
                and decision["finding"] is not None
                and not _valid_finding(decision["finding"])
            )
            or (
                "ledger" in decision
                and not _valid_ledger(decision["ledger"])
            )
        ):
            self.contract_errors += 1
            return {"action": {"type": "WAIT"}, "memory": "", "invalid": True}
        finding = decision.get("finding")
        if finding is not None:
            self.finding = copy.deepcopy(finding)
        envelope = {
            "action": copy.deepcopy(decision["action"]),
            "finding": copy.deepcopy(finding),
            "memory": "",
        }
        if finding is None:
            envelope.pop("finding")
        if "ledger" in decision:
            envelope["ledger"] = copy.deepcopy(decision["ledger"])
        return envelope

    def after_step(
        self, post_observation: dict[str, Any] | None, result: dict[str, Any],
    ) -> None:
        del post_observation
        self._agent.observe_result(_detached_json(result, path="action result"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._agent.close()


def load_agent_factory(reference: str) -> AgentFactory:
    """Load one exact ``module:function`` participant factory."""

    if not isinstance(reference, str) or _FACTORY_REFERENCE.fullmatch(reference) is None:
        raise AgentContractError("agent reference must use exact module:function syntax")
    module_name, attribute = reference.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ValueError) as exc:
        raise AgentContractError(f"agent module {module_name!r} could not be imported") from exc
    if not hasattr(module, attribute):
        raise AgentContractError(f"agent factory {reference!r} does not exist")
    factory = getattr(module, attribute)
    if not callable(factory):
        raise AgentContractError(f"agent factory {reference!r} is not callable")
    return factory


def run_agent_episode(
    world: Any,
    factory: AgentFactory,
    context: dict[str, Any],
    *,
    name: str,
    capture: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, str]]:
    """Run one fresh participant agent without changing legacy trace codecs."""

    from .experiment import run_episode

    adapter = AgentPolicyAdapter(factory, context, name=name)
    try:
        result, trace = run_episode(world, adapter, capture=capture)
        return result, trace, copy.deepcopy(adapter.finding)
    finally:
        adapter.close()


__all__ = [
    "AgentContractError",
    "AgentFactory",
    "AgentPolicyAdapter",
    "EVIDENCE_LEDGER_SCHEMA",
    "WorldZeroAgent",
    "agent_context",
    "load_agent_factory",
    "run_agent_episode",
]
