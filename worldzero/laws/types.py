"""Immutable public records for WorldZero law-family plugins.

The kernel and plugins exchange only these detached records.  JSON-bearing
fields are copied and recursively frozen on construction so callers cannot
retain a mutable alias into evaluator state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeAlias


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | Mapping[str, "JSONValue"] | Sequence["JSONValue"]
FrozenJSONValue: TypeAlias = JSONScalar | Mapping[str, "FrozenJSONValue"] | tuple["FrozenJSONValue", ...]

SUPPORTED_API_VERSION = "1.0"

_FAMILY_ID = re.compile(
    r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*:[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$"
)
_STABLE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._:-][a-z0-9]+)*$")
_SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def _require_nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_family_id(value: object) -> str:
    if not isinstance(value, str) or _FAMILY_ID.fullmatch(value) is None:
        raise ValueError("family_id must be a lower-case namespaced identifier")
    return value


def _require_stable_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _STABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable lower-case identifier")
    return value


def _require_semver(name: str, value: object) -> str:
    if not isinstance(value, str) or _SEMVER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a semantic version")
    return value


def _require_index(name: str, value: object, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        suffix = f" in 0..{maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a nonnegative integer{suffix}")
    return value


def _require_finite(name: str, value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def freeze_json(value: JSONValue, *, path: str = "value") -> FrozenJSONValue:
    """Validate, detach, and recursively freeze one JSON-compatible value."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJSONValue] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"{path} must contain only string JSON object keys")
            frozen[key] = freeze_json(value[key], path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path} must contain only JSON-compatible values")


def thaw_json(value: FrozenJSONValue) -> JSONValue:
    """Return a detached ordinary dict/list representation for persistence."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _reject_kernel_reserved_namespace(
    value: FrozenJSONValue, *, path: str,
) -> None:
    """Keep kernel-owned persistence names out of every plugin JSON subtree."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.startswith("_worldzero"):
                raise ValueError(f"{path}.{key} uses the kernel reserved namespace")
            _reject_kernel_reserved_namespace(item, path=f"{path}.{key}")
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            _reject_kernel_reserved_namespace(item, path=f"{path}[{index}]")


def _validate_closed_schema(schema: object, *, path: str = "observation_schema") -> None:
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path} must be a JSON schema object")
    schema_type = schema.get("type")
    allowed_keywords = {
        "object": frozenset({"type", "additionalProperties", "properties", "required"}),
        "array": frozenset({"type", "items"}),
        "null": frozenset({"type"}),
        "boolean": frozenset({"type"}),
        "integer": frozenset({"type"}),
        "number": frozenset({"type"}),
        "string": frozenset({"type"}),
    }
    allowed = allowed_keywords.get(schema_type, frozenset({"type"}))
    unexpected = sorted(key for key in schema if key not in allowed)
    if unexpected:
        raise ValueError(f"{path}.{unexpected[0]} is not supported by the closed observation schema")
    if schema_type == "object":
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"{path}.additionalProperties must be false")
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path}.properties must be an object")
        for name, subschema in properties.items():
            if not isinstance(name, str):
                raise ValueError(f"{path}.properties keys must be strings")
            _validate_closed_schema(subschema, path=f"{path}.properties.{name}")
        required = schema.get("required", ())
        if not isinstance(required, (list, tuple)) or any(not isinstance(item, str) for item in required):
            raise ValueError(f"{path}.required must be an array of property names")
        if len(set(required)) != len(required) or not set(required).issubset(properties):
            raise ValueError(f"{path}.required must contain unique declared properties")
    elif schema_type == "array":
        if "items" not in schema:
            raise ValueError(f"{path}.items is required for arrays")
        _validate_closed_schema(schema["items"], path=f"{path}.items")
    elif schema_type not in {"null", "boolean", "integer", "number", "string"}:
        raise ValueError(f"{path}.type is not supported by the closed observation schema")


@dataclass(frozen=True)
class FamilyDescriptor:
    """Stable identity and declared capabilities for one family implementation."""

    family_id: str
    api_version: str
    family_version: str
    display_name: str
    package: str
    package_version: str
    capabilities: frozenset[str]
    observation_schema: Mapping[str, FrozenJSONValue]
    documentation_url: str | None = None

    def __post_init__(self) -> None:
        _require_family_id(self.family_id)
        if self.api_version != SUPPORTED_API_VERSION:
            raise ValueError(f"api_version must be {SUPPORTED_API_VERSION}")
        _require_semver("family_version", self.family_version)
        _require_nonempty("display_name", self.display_name)
        _require_nonempty("package", self.package)
        _require_semver("package_version", self.package_version)
        try:
            capabilities = frozenset(self.capabilities)
        except TypeError as exc:
            raise TypeError("capabilities must be an iterable of identifiers") from exc
        if any(_STABLE_ID.fullmatch(item) is None for item in capabilities if isinstance(item, str)) or any(
            not isinstance(item, str) for item in capabilities
        ):
            raise ValueError("capabilities must contain stable lower-case identifiers")
        _validate_closed_schema(self.observation_schema)
        if self.observation_schema.get("type") != "object":
            raise ValueError("observation_schema.type must be object at the projection root")
        frozen_schema = freeze_json(self.observation_schema, path="observation_schema")
        if not isinstance(frozen_schema, Mapping):  # pragma: no cover - guaranteed above
            raise TypeError("observation_schema must be a JSON object")
        if self.documentation_url is not None:
            _require_nonempty("documentation_url", self.documentation_url)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "observation_schema", frozen_schema)

    def persistence_dict(self) -> dict[str, JSONValue]:
        """Return the deterministic JSON representation frozen into run identity."""

        return {
            "api_version": self.api_version,
            "capabilities": sorted(self.capabilities),
            "display_name": self.display_name,
            "documentation_url": self.documentation_url,
            "family_id": self.family_id,
            "family_version": self.family_version,
            "observation_schema": thaw_json(self.observation_schema),
            "package": self.package,
            "package_version": self.package_version,
        }


@dataclass(frozen=True)
class FamilyInstance:
    """Sampled hidden parameters and plugin-private JSON state."""

    family_id: str
    family_version: str
    hidden_parameters: Mapping[str, FrozenJSONValue]
    private_state: Mapping[str, FrozenJSONValue]
    enabled: bool = True

    def __post_init__(self) -> None:
        _require_family_id(self.family_id)
        _require_semver("family_version", self.family_version)
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        hidden = freeze_json(self.hidden_parameters, path="hidden_parameters")
        state = freeze_json(self.private_state, path="private_state")
        if not isinstance(hidden, Mapping) or not isinstance(state, Mapping):
            raise TypeError("hidden_parameters and private_state must be JSON objects")
        _reject_kernel_reserved_namespace(hidden, path="hidden_parameters")
        _reject_kernel_reserved_namespace(state, path="private_state")
        object.__setattr__(self, "hidden_parameters", hidden)
        object.__setattr__(self, "private_state", state)


@dataclass(frozen=True)
class SampleContext:
    """Kernel-supplied deterministic named draws; it exposes no ambient RNG."""

    named_draws: Mapping[str, FrozenJSONValue]
    named_seeds: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        draws = freeze_json(self.named_draws, path="named_draws")
        if not isinstance(draws, Mapping):
            raise TypeError("named_draws must be a JSON object")
        seeds = dict(self.named_seeds)
        if any(not isinstance(name, str) or type(seed) is not int or seed < 0 for name, seed in seeds.items()):
            raise ValueError("named_seeds must map names to nonnegative integers")
        object.__setattr__(self, "named_draws", draws)
        object.__setattr__(self, "named_seeds", MappingProxyType(dict(sorted(seeds.items()))))

    def draw(self, name: str) -> FrozenJSONValue:
        return self.named_draws[name]

    def sample_indices(
        self, name: str, *, population_size: int, count: int,
    ) -> tuple[int, ...]:
        """Sample a named, replayable subset without exposing an RNG object."""

        _require_index("population_size", population_size)
        _require_index("count", count)
        if population_size == 0 or count > population_size:
            raise ValueError("count must not exceed a positive population_size")
        import numpy as np

        values = np.random.default_rng(self.named_seeds[name]).choice(
            population_size, count, replace=False,
        )
        return tuple(sorted(int(value) for value in values))


class TargetDomain(str, Enum):
    CELL = "cell"
    MODULE = "module"
    GLOBAL = "global"


class DrawRequirement(str, Enum):
    TARGET_INDEX = "target_index"
    ACCEPTANCE_UNIFORM = "acceptance_uniform"


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    envelope_rate: float
    target_domain: TargetDomain
    draw_requirements: tuple[DrawRequirement, ...] = ()

    def __post_init__(self) -> None:
        _require_stable_id("channel_id", self.channel_id)
        object.__setattr__(self, "envelope_rate", _require_finite("envelope_rate", self.envelope_rate, nonnegative=True))
        try:
            domain = TargetDomain(self.target_domain)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_domain is invalid") from exc
        try:
            requirements = tuple(DrawRequirement(item) for item in self.draw_requirements)
        except (TypeError, ValueError) as exc:
            raise ValueError("draw requirement is invalid") from exc
        if len(set(requirements)) != len(requirements):
            raise ValueError("draw_requirements contain a duplicate")
        if domain in {TargetDomain.CELL, TargetDomain.MODULE} and DrawRequirement.TARGET_INDEX not in requirements:
            raise ValueError("target_index draw requirement is required for indexed target domains")
        if domain is TargetDomain.GLOBAL and DrawRequirement.TARGET_INDEX in requirements:
            raise ValueError("target_index draw requirement is invalid for the global target domain")
        object.__setattr__(self, "target_domain", domain)
        object.__setattr__(self, "draw_requirements", requirements)


def validate_channel_specs(channels: Sequence[ChannelSpec]) -> tuple[ChannelSpec, ...]:
    """Validate unique channel IDs and return canonical ID order."""

    result = tuple(channels)
    if any(not isinstance(channel, ChannelSpec) for channel in result):
        raise TypeError("channels must contain only ChannelSpec records")
    ids = [channel.channel_id for channel in result]
    if len(set(ids)) != len(ids):
        raise ValueError("channel IDs contain a duplicate")
    return tuple(sorted(result, key=lambda channel: channel.channel_id))


@dataclass(frozen=True)
class ProposalDraw:
    channel_id: str
    target_index: int
    acceptance_uniform: float
    proposal_index: int
    simulated_time: float

    def __post_init__(self) -> None:
        _require_stable_id("channel_id", self.channel_id)
        _require_index("target_index", self.target_index)
        _require_index("proposal_index", self.proposal_index)
        uniform = _require_finite("acceptance_uniform", self.acceptance_uniform)
        if not 0.0 <= uniform < 1.0:
            raise ValueError("acceptance_uniform must be in [0, 1)")
        object.__setattr__(self, "acceptance_uniform", uniform)
        object.__setattr__(self, "simulated_time", _require_finite("simulated_time", self.simulated_time, nonnegative=True))


def _freeze_position(name: str, position: Sequence[int] | None) -> tuple[int, int] | None:
    if position is None:
        return None
    if len(position) != 2:
        raise ValueError(f"{name} position must contain row and column")
    result = (_require_index(f"{name} position row", position[0]), _require_index(f"{name} position column", position[1]))
    return result


def _freeze_grid(name: str, values: Sequence[Sequence[int]], width: int, height: int) -> tuple[tuple[int, ...], ...]:
    grid = tuple(tuple(row) for row in values)
    if len(grid) != height or any(len(row) != width for row in grid):
        raise ValueError(f"{name} dimensions do not match width and height")
    if any(type(item) is not int for row in grid for item in row):
        raise TypeError(f"{name} must contain integers")
    return grid


@dataclass(frozen=True)
class PublicSubstrateView:
    """Read-only locally observable substrate state supplied to projection."""

    width: int
    height: int
    agent_position: tuple[int, int]
    module_positions: tuple[tuple[int, int] | None, ...]
    module_states: tuple[Mapping[str, FrozenJSONValue], ...]
    resources: tuple[tuple[int, ...], ...]
    terrain: tuple[tuple[int, ...], ...]
    simulated_time: float

    def __post_init__(self) -> None:
        width = _require_index("width", self.width)
        height = _require_index("height", self.height)
        if width == 0 or height == 0:
            raise ValueError("width and height must be positive")
        object.__setattr__(self, "agent_position", _freeze_position("agent", self.agent_position))
        positions = tuple(_freeze_position("module", item) for item in self.module_positions)
        states: list[Mapping[str, FrozenJSONValue]] = []
        for index, item in enumerate(self.module_states):
            frozen = freeze_json(item, path=f"module_states[{index}]")
            if not isinstance(frozen, Mapping):
                raise TypeError("module_states entries must be JSON objects")
            states.append(frozen)
        if len(positions) != len(states):
            raise ValueError("module positions and states must have equal lengths")
        object.__setattr__(self, "module_positions", positions)
        object.__setattr__(self, "module_states", tuple(states))
        object.__setattr__(self, "resources", _freeze_grid("resources", self.resources, width, height))
        object.__setattr__(self, "terrain", _freeze_grid("terrain", self.terrain, width, height))
        object.__setattr__(self, "simulated_time", _require_finite("simulated_time", self.simulated_time, nonnegative=True))


@dataclass(frozen=True)
class SubstrateView(PublicSubstrateView):
    """Evaluator substrate view with public-independent kernel counters."""

    kernel_counters: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        counters = dict(self.kernel_counters)
        if any(not isinstance(key, str) or type(value) is not int or value < 0 for key, value in counters.items()):
            raise ValueError("kernel_counters must map names to nonnegative integers")
        object.__setattr__(self, "kernel_counters", MappingProxyType(dict(sorted(counters.items()))))


@dataclass(frozen=True)
class DerivedLawState:
    state: Mapping[str, FrozenJSONValue]
    functional: bool
    affected_locations: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        frozen = freeze_json(self.state, path="derived_state")
        if not isinstance(frozen, Mapping):
            raise TypeError("state must be a JSON object")
        if type(self.functional) is not bool:
            raise TypeError("functional must be a boolean")
        locations = tuple(self.affected_locations)
        for location in locations:
            _require_index("affected_locations", location)
        if len(set(locations)) != len(locations):
            raise ValueError("affected_locations contain a duplicate")
        object.__setattr__(self, "state", frozen)
        object.__setattr__(self, "affected_locations", locations)


@dataclass(frozen=True)
class ResourceReplacement:
    cell_index: int
    expected_value: int
    replacement_value: int
    required_capability = "resource_transition"

    def __post_init__(self) -> None:
        _require_index("cell_index", self.cell_index)
        _require_index("expected_value", self.expected_value)
        _require_index("replacement_value", self.replacement_value)


@dataclass(frozen=True)
class ResourcePreservation:
    cell_index: int
    expected_value: int
    required_capability = "resource_preservation"

    def __post_init__(self) -> None:
        _require_index("cell_index", self.cell_index)
        _require_index("expected_value", self.expected_value)


@dataclass(frozen=True)
class ModulePositionChange:
    module_index: int
    expected_position: tuple[int, int] | None
    replacement_position: tuple[int, int] | None
    required_capability = "geometry_control"

    def __post_init__(self) -> None:
        _require_index("module_index", self.module_index, maximum=2)
        object.__setattr__(self, "expected_position", _freeze_position("expected", self.expected_position))
        object.__setattr__(self, "replacement_position", _freeze_position("replacement", self.replacement_position))


@dataclass(frozen=True)
class ModuleStateChange:
    module_index: int
    state_key: str
    expected_state: FrozenJSONValue
    replacement_state: FrozenJSONValue
    required_capability = "module_transition"

    def __post_init__(self) -> None:
        _require_index("module_index", self.module_index, maximum=2)
        _require_stable_id("state_key", self.state_key)
        object.__setattr__(self, "expected_state", freeze_json(self.expected_state, path="expected_state"))
        object.__setattr__(self, "replacement_state", freeze_json(self.replacement_state, path="replacement_state"))


@dataclass(frozen=True)
class EventEvidence:
    event_type: str
    location_index: int | None = None
    details: Mapping[str, FrozenJSONValue] = field(default_factory=dict)
    required_capability = "event_evidence"

    def __post_init__(self) -> None:
        _require_stable_id("event_type", self.event_type)
        if self.location_index is not None:
            _require_index("location_index", self.location_index)
        details = freeze_json(self.details, path="event_details")
        if not isinstance(details, Mapping):
            raise TypeError("details must be a JSON object")
        object.__setattr__(self, "details", details)


TransitionOperation: TypeAlias = (
    ResourceReplacement | ResourcePreservation | ModulePositionChange | ModuleStateChange | EventEvidence
)
_TRANSITION_OPERATIONS = (
    ResourceReplacement,
    ResourcePreservation,
    ModulePositionChange,
    ModuleStateChange,
    EventEvidence,
)


@dataclass(frozen=True)
class AccountingDelta:
    material_delta: int = 0
    energy_delta: float = 0.0

    def __post_init__(self) -> None:
        if type(self.material_delta) is not int:
            raise TypeError("material_delta must be an integer")
        object.__setattr__(self, "energy_delta", _require_finite("energy_delta", self.energy_delta))


def _validate_operations(
    operations: Sequence[TransitionOperation], declared_capabilities: frozenset[str],
) -> tuple[TransitionOperation, ...]:
    result = tuple(operations)
    if any(not isinstance(operation, _TRANSITION_OPERATIONS) for operation in result):
        raise TypeError("operations contain an undeclared transition operation type")
    missing = sorted(
        {operation.required_capability for operation in result} - declared_capabilities
    )
    if missing:
        raise ValueError(f"transition uses undeclared capabilities: {', '.join(missing)}")
    return result


@dataclass(frozen=True)
class LawTransition:
    operations: tuple[TransitionOperation, ...]
    accounting: AccountingDelta
    declared_capabilities: frozenset[str]

    def __post_init__(self) -> None:
        capabilities = frozenset(self.declared_capabilities)
        for capability in capabilities:
            _require_stable_id("declared capability", capability)
        if not isinstance(self.accounting, AccountingDelta):
            raise TypeError("accounting must be an AccountingDelta")
        object.__setattr__(self, "operations", _validate_operations(self.operations, capabilities))
        object.__setattr__(self, "declared_capabilities", capabilities)


@dataclass(frozen=True)
class KernelProposalRejection:
    """A closed request to reject one otherwise-applicable kernel RAW decay.

    The record cannot carry accounting or substrate replacement operations: it
    can only prove that the current RAW value is preserved and emit detached
    evaluator evidence.
    """

    proposal: ProposalDraw
    operations: tuple[ResourcePreservation | EventEvidence, ...]
    declared_capabilities: frozenset[str]
    accounting: AccountingDelta = field(default_factory=AccountingDelta, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, ProposalDraw) or self.proposal.channel_id != "raw_decay":
            raise ValueError("proposal rejection is restricted to the raw_decay kernel-owned channel")
        capabilities = frozenset(self.declared_capabilities)
        if "proposal_filter" not in capabilities:
            raise ValueError("proposal rejection requires proposal_filter capability")
        operations = tuple(self.operations)
        if any(not isinstance(item, (ResourcePreservation, EventEvidence)) for item in operations):
            raise ValueError("proposal rejection may only preserve a resource and emit evidence")
        validated = _validate_operations(operations, capabilities)
        preservations = [item for item in validated if isinstance(item, ResourcePreservation)]
        if len(preservations) != 1 or preservations[0].cell_index != self.proposal.target_index:
            raise ValueError("proposal rejection must preserve exactly its target")
        object.__setattr__(self, "operations", validated)
        object.__setattr__(self, "declared_capabilities", capabilities)


@dataclass(frozen=True)
class PrivateStateTransition:
    """One immutable, explicitly capability-gated family-private state update."""

    expected_state: Mapping[str, FrozenJSONValue]
    replacement_state: Mapping[str, FrozenJSONValue]
    declared_capabilities: frozenset[str]

    def __post_init__(self) -> None:
        capabilities = frozenset(self.declared_capabilities)
        if capabilities != frozenset({"private_state_transition"}):
            raise ValueError("private state update requires only private_state_transition capability")
        expected = freeze_json(self.expected_state, path="expected_private_state")
        replacement = freeze_json(self.replacement_state, path="replacement_private_state")
        if not isinstance(expected, Mapping) or not isinstance(replacement, Mapping):
            raise TypeError("private states must be JSON objects")
        _reject_kernel_reserved_namespace(expected, path="expected_private_state")
        _reject_kernel_reserved_namespace(replacement, path="replacement_private_state")
        object.__setattr__(self, "expected_state", expected)
        object.__setattr__(self, "replacement_state", replacement)
        object.__setattr__(self, "declared_capabilities", capabilities)

    def resulting_instance(self, instance: FamilyInstance) -> FamilyInstance:
        """Apply this transition to one exact expected immutable instance."""

        if not isinstance(instance, FamilyInstance) or instance.private_state != self.expected_state:
            raise ValueError("private state transition does not match the current instance")
        return FamilyInstance(
            instance.family_id, instance.family_version, instance.hidden_parameters,
            self.replacement_state, instance.enabled,
        )


class ControlKind(str, Enum):
    NULL = "null"
    KNOCKOUT = "knockout"
    BROKEN = "broken"
    RETAINED = "retained"


@dataclass(frozen=True)
class ControlSpec:
    kind: ControlKind
    matching_constraints: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        try:
            kind = ControlKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("control kind is invalid") from exc
        constraints = frozenset(self.matching_constraints)
        for constraint in constraints:
            _require_stable_id("matching constraint", constraint)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "matching_constraints", constraints)


@dataclass(frozen=True)
class ControlSuite:
    null: ControlSpec
    knockout: ControlSpec
    broken: ControlSpec
    retained: ControlSpec

    def __post_init__(self) -> None:
        expected = (
            ("null", self.null, ControlKind.NULL),
            ("knockout", self.knockout, ControlKind.KNOCKOUT),
            ("broken", self.broken, ControlKind.BROKEN),
            ("retained", self.retained, ControlKind.RETAINED),
        )
        for field_name, spec, kind in expected:
            if not isinstance(spec, ControlSpec) or spec.kind is not kind:
                raise ValueError(f"{field_name} must contain the {kind.value} control")


@dataclass(frozen=True)
class InterventionTransition:
    control: ControlKind
    operations: tuple[TransitionOperation, ...]
    accounting: AccountingDelta
    declared_capabilities: frozenset[str]
    resulting_instance: FamilyInstance

    def __post_init__(self) -> None:
        try:
            control = ControlKind(self.control)
        except (TypeError, ValueError) as exc:
            raise ValueError("control is invalid") from exc
        if not isinstance(self.accounting, AccountingDelta):
            raise TypeError("accounting must be an AccountingDelta")
        if not isinstance(self.resulting_instance, FamilyInstance):
            raise TypeError("resulting_instance must be a FamilyInstance")
        capabilities = frozenset(self.declared_capabilities)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "operations", _validate_operations(self.operations, capabilities))
        object.__setattr__(self, "declared_capabilities", capabilities)


@dataclass(frozen=True)
class EvaluatorTrace:
    events: tuple[Mapping[str, FrozenJSONValue], ...]
    terminal: Mapping[str, FrozenJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        events: list[Mapping[str, FrozenJSONValue]] = []
        for index, event in enumerate(self.events):
            frozen = freeze_json(event, path=f"events[{index}]")
            if not isinstance(frozen, Mapping):
                raise TypeError("events must contain JSON objects")
            events.append(frozen)
        terminal = freeze_json(self.terminal, path="terminal")
        if not isinstance(terminal, Mapping):
            raise TypeError("terminal must be a JSON object")
        object.__setattr__(self, "events", tuple(events))
        object.__setattr__(self, "terminal", terminal)


_EVIDENCE_ORIGINS = {"model_placement", "model_drop", "death_drop", "pre_existing", "none"}


@dataclass(frozen=True)
class FamilyEvidence:
    stage_evidence: Mapping[str, FrozenJSONValue]
    event_references: tuple[int, ...] = ()
    origin: str = "none"
    structure_constructed: bool = False
    function_observed: bool = False
    effect_observed: bool = False
    relevant_consequence_observed: bool = False
    intervention_preceded_consequence: bool = False
    discriminating_verification: bool = False
    retained_or_reconstructed: bool = False
    linked_benefit: bool = False
    diagnostics: Mapping[str, FrozenJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stages = freeze_json(self.stage_evidence, path="stage_evidence")
        diagnostics = freeze_json(self.diagnostics, path="diagnostics")
        if not isinstance(stages, Mapping) or not isinstance(diagnostics, Mapping):
            raise TypeError("stage_evidence and diagnostics must be JSON objects")
        references = tuple(self.event_references)
        for reference in references:
            _require_index("event_references", reference)
        if len(set(references)) != len(references):
            raise ValueError("event_references contain a duplicate")
        if self.origin not in _EVIDENCE_ORIGINS:
            raise ValueError("origin is invalid")
        for name in (
            "structure_constructed",
            "function_observed",
            "effect_observed",
            "relevant_consequence_observed",
            "intervention_preceded_consequence",
            "discriminating_verification",
            "retained_or_reconstructed",
            "linked_benefit",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(self, "stage_evidence", stages)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "event_references", references)

    def persistence_dict(self, *, include_diagnostics: bool = True) -> dict[str, JSONValue]:
        """Return the deterministic JSON persistence representation."""

        result: dict[str, JSONValue] = {
            "stage_evidence": thaw_json(self.stage_evidence),
            "event_references": list(self.event_references),
            "origin": self.origin,
            "structure_constructed": self.structure_constructed,
            "function_observed": self.function_observed,
            "effect_observed": self.effect_observed,
            "relevant_consequence_observed": self.relevant_consequence_observed,
            "intervention_preceded_consequence": self.intervention_preceded_consequence,
            "discriminating_verification": self.discriminating_verification,
            "retained_or_reconstructed": self.retained_or_reconstructed,
            "linked_benefit": self.linked_benefit,
        }
        if include_diagnostics:
            result["diagnostics"] = thaw_json(self.diagnostics)
        return result

    @classmethod
    def from_persistence(cls, value: Mapping[str, JSONValue]) -> "FamilyEvidence":
        """Validate and reconstruct an exact persisted evidence record."""

        required = {
            "stage_evidence", "event_references", "origin", "structure_constructed",
            "function_observed", "effect_observed", "relevant_consequence_observed",
            "intervention_preceded_consequence", "discriminating_verification",
            "retained_or_reconstructed", "linked_benefit", "diagnostics",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("FamilyEvidence persistence fields are invalid")
        references = value["event_references"]
        if not isinstance(references, (list, tuple)):
            raise TypeError("FamilyEvidence event_references must be an array")
        return cls(
            stage_evidence=value["stage_evidence"],  # type: ignore[arg-type]
            event_references=tuple(references),  # type: ignore[arg-type]
            origin=value["origin"],  # type: ignore[arg-type]
            structure_constructed=value["structure_constructed"],  # type: ignore[arg-type]
            function_observed=value["function_observed"],  # type: ignore[arg-type]
            effect_observed=value["effect_observed"],  # type: ignore[arg-type]
            relevant_consequence_observed=value["relevant_consequence_observed"],  # type: ignore[arg-type]
            intervention_preceded_consequence=value["intervention_preceded_consequence"],  # type: ignore[arg-type]
            discriminating_verification=value["discriminating_verification"],  # type: ignore[arg-type]
            retained_or_reconstructed=value["retained_or_reconstructed"],  # type: ignore[arg-type]
            linked_benefit=value["linked_benefit"],  # type: ignore[arg-type]
            diagnostics=value["diagnostics"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    kind: str
    expected: FrozenJSONValue
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    samples: int = 1
    parameters: Mapping[str, FrozenJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_stable_id("case_id", self.case_id)
        _require_stable_id("kind", self.kind)
        object.__setattr__(self, "expected", freeze_json(self.expected, path="expected"))
        absolute = _require_finite("absolute_tolerance", self.absolute_tolerance, nonnegative=True)
        relative = _require_finite("relative_tolerance", self.relative_tolerance, nonnegative=True)
        if type(self.samples) is not int or self.samples <= 0:
            raise ValueError("samples must be a positive integer")
        parameters = freeze_json(self.parameters, path="parameters")
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a JSON object")
        object.__setattr__(self, "absolute_tolerance", absolute)
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "parameters", parameters)


__all__ = [
    "AccountingDelta",
    "CalibrationCase",
    "ChannelSpec",
    "ControlKind",
    "ControlSpec",
    "ControlSuite",
    "DerivedLawState",
    "DrawRequirement",
    "EvaluatorTrace",
    "EventEvidence",
    "FamilyDescriptor",
    "FamilyEvidence",
    "FamilyInstance",
    "FrozenJSONValue",
    "InterventionTransition",
    "JSONValue",
    "KernelProposalRejection",
    "LawTransition",
    "ModulePositionChange",
    "ModuleStateChange",
    "ProposalDraw",
    "PrivateStateTransition",
    "PublicSubstrateView",
    "ResourcePreservation",
    "ResourceReplacement",
    "SUPPORTED_API_VERSION",
    "SampleContext",
    "SubstrateView",
    "TargetDomain",
    "TransitionOperation",
    "freeze_json",
    "thaw_json",
    "validate_channel_specs",
]
