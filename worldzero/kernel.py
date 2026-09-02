"""Fixed continuous-time stochastic kernel with typed law-family callbacks.

Environmental proposals form a STATE-INDEPENDENT Poisson stream. A proposal
fires only if its local preconditions hold (uniformization/thinning). Cloned
worlds therefore receive identical proposed shocks even after interventions.
There are no fixed simulation ticks and no hierarchy or task-specific tools.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from collections.abc import Mapping
from typing import Any
import bisect
import copy
import json
import math
import re

import numpy as np

from .laws.registry import RegisteredFamily, calibration_suite_fingerprint, resolve_family
from .laws.types import (
    AccountingDelta,
    ChannelSpec,
    ControlSuite,
    ControlKind,
    DerivedLawState,
    DrawRequirement,
    EventEvidence,
    EvaluatorTrace,
    FamilyEvidence,
    FamilyInstance,
    InterventionTransition,
    KernelProposalRejection,
    LawTransition,
    ModulePositionChange,
    ModuleStateChange,
    ProposalDraw,
    PrivateStateTransition,
    PublicSubstrateView,
    ResourcePreservation,
    ResourceReplacement,
    SampleContext,
    SubstrateView,
    TargetDomain,
    freeze_json,
    thaw_json,
    validate_channel_specs,
)
from .util import canonical, derive_seed, digest, require_expected_sha256, require_finite

EMPTY, RAW, RICH = 0, 1, 2
DIRECTIONS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
PLUGIN_PROPOSAL_RECORD_LIMIT = 100_000
PLUGIN_PRIVATE_TRANSITION_RECORD_LIMIT = 1_024


def encode_transition_operation(operation: object) -> dict[str, Any]:
    """Encode one closed transition operation as JSON, never as a plugin object."""

    if isinstance(operation, ResourceReplacement):
        return {"type": "resource_replacement", "cell_index": operation.cell_index,
                "expected_value": operation.expected_value,
                "replacement_value": operation.replacement_value}
    if isinstance(operation, ResourcePreservation):
        return {"type": "resource_preservation", "cell_index": operation.cell_index,
                "expected_value": operation.expected_value}
    if isinstance(operation, ModulePositionChange):
        return {"type": "module_position_change", "module_index": operation.module_index,
                "expected_position": list(operation.expected_position) if operation.expected_position is not None else None,
                "replacement_position": list(operation.replacement_position) if operation.replacement_position is not None else None}
    if isinstance(operation, ModuleStateChange):
        return {"type": "module_state_change", "module_index": operation.module_index,
                "state_key": operation.state_key,
                "expected_state": thaw_json(operation.expected_state),
                "replacement_state": thaw_json(operation.replacement_state)}
    if isinstance(operation, EventEvidence):
        return {"type": "event_evidence", "event_type": operation.event_type,
                "location_index": operation.location_index,
                "details": thaw_json(operation.details)}
    raise TypeError("transition operation type is not in the closed codec")


def decode_transition_operation(value: object) -> object:
    """Validate and decode one operation from the explicit persistence codec."""

    if not isinstance(value, dict) or type(value.get("type")) is not str:
        raise ValueError("transition operation codec is invalid")
    operation_type = value["type"]
    fields = set(value)
    if operation_type == "resource_replacement" and fields == {
        "type", "cell_index", "expected_value", "replacement_value",
    }:
        return ResourceReplacement(value["cell_index"], value["expected_value"], value["replacement_value"])
    if operation_type == "resource_preservation" and fields == {
        "type", "cell_index", "expected_value",
    }:
        return ResourcePreservation(value["cell_index"], value["expected_value"])
    if operation_type == "module_position_change" and fields == {
        "type", "module_index", "expected_position", "replacement_position",
    }:
        return ModulePositionChange(value["module_index"], value["expected_position"], value["replacement_position"])
    if operation_type == "module_state_change" and fields == {
        "type", "module_index", "state_key", "expected_state", "replacement_state",
    }:
        return ModuleStateChange(value["module_index"], value["state_key"], value["expected_state"], value["replacement_state"])
    if operation_type == "event_evidence" and fields == {
        "type", "event_type", "location_index", "details",
    }:
        return EventEvidence(value["event_type"], value["location_index"], value["details"])
    raise ValueError("transition operation codec fields are invalid")


class FamilyCallbackError(RuntimeError):
    """A trusted family callback failed; execution never falls back."""


class FamilyTransitionError(RuntimeError):
    """A family returned a transition the kernel cannot apply atomically."""


def _validate_projection(value: Any, schema: Mapping[str, Any], *, path: str = "law_observation") -> None:
    """Validate the deliberately small closed schema subset admitted by descriptors."""

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be an object")
        properties = schema["properties"]
        required = set(schema.get("required", ()))
        if not set(value).issubset(properties) or not required.issubset(value):
            raise ValueError(f"{path} violates the declared object fields")
        for key, item in value.items():
            _validate_projection(item, properties[key], path=f"{path}.{key}")
    elif schema_type == "array":
        if not isinstance(value, tuple):
            raise TypeError(f"{path} must be an array")
        for index, item in enumerate(value):
            _validate_projection(item, schema["items"], path=f"{path}[{index}]")
    elif schema_type == "null":
        if value is not None:
            raise TypeError(f"{path} must be null")
    elif schema_type == "boolean":
        if type(value) is not bool:
            raise TypeError(f"{path} must be a boolean")
    elif schema_type == "integer":
        if type(value) is not int:
            raise TypeError(f"{path} must be an integer")
    elif schema_type == "number":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise TypeError(f"{path} must be a finite number")
    elif schema_type == "string":
        if type(value) is not str:
            raise TypeError(f"{path} must be a string")


@dataclass(frozen=True)
class Config:
    width: int = 13
    height: int = 9
    radius: int = 3
    lifespan: float = 160.0
    initial_energy: float = 22.0
    metabolism: float = 0.24
    cognition_energy: float = 0.06
    cognition_time: float = 0.20
    move_time: float = 0.8
    manipulate_time: float = 0.6
    consume_time: float = 0.25
    default_wait: float = 2.0
    max_wait: float = 8.0
    raw_energy: float = 0.8
    rich_energy: float = 6.0
    source_rate: float = 0.012
    raw_decay: float = 0.008
    rich_decay: float = 0.016
    conversion_rate: float = 0.45
    module_decay: float = 0.00010
    regime_rate: float = 0.004
    lean_source_multiplier: float = 0.6
    initial_resource_fraction: float = 0.32
    private_memory_chars: int = 2400
    max_decisions: int = 500

    def __post_init__(self) -> None:
        for name in ("width", "height", "radius", "private_memory_chars", "max_decisions"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width < 9 or self.height < 7:
            raise ValueError("The generator requires width >= 9 and height >= 7")
        for name in ("lifespan", "initial_energy", "cognition_time", "move_time", "manipulate_time",
                     "consume_time", "default_wait", "max_wait", "raw_energy", "rich_energy"):
            require_finite(name, getattr(self, name), positive=True)
        for name in ("metabolism", "cognition_energy", "source_rate", "raw_decay", "rich_decay",
                     "conversion_rate", "module_decay", "regime_rate"):
            require_finite(name, getattr(self, name))
        for name in ("lean_source_multiplier", "initial_resource_fraction"):
            require_finite(name, getattr(self, name))
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must not exceed 1")
        if self.rich_energy <= self.raw_energy or self.max_wait < self.default_wait:
            raise ValueError("Invalid energy or waiting-time ordering")


@dataclass(frozen=True)
class Law:
    pair: tuple[int, int]
    family: str = "catalysis"
    geometry: str = "adjacent"

    def __post_init__(self) -> None:
        if len(self.pair) != 2 or len(set(self.pair)) != 2 or any(type(x) is not int or x not in range(3) for x in self.pair):
            raise ValueError("pair must contain two different component classes in 0..2")
        if self.family not in {"catalysis", "null"}:
            raise ValueError("Unknown law family")
        if self.geometry not in {"adjacent", "distance2"}:
            raise ValueError("Unknown geometry")


@dataclass
class Agent:
    position: tuple[int, int]
    energy: float
    born: float
    generation: int = 1
    age: float = 0.0
    alive: bool = True
    termination: str | None = None
    inventory: int | None = None
    decisions: int = 0
    invalid_actions: int = 0
    raw_consumed: int = 0
    rich_consumed: int = 0
    memory: str = ""
    last_result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PluginLawView:
    """Evaluator-only compatibility shape for non-legacy plugin families."""

    pair: tuple[int, int]
    family: str
    geometry: str


class World:
    """The only policy-facing methods are observe() and step(JSON decision).

    The runner serializes observations before calling a policy; it NEVER passes
    a World object. Diagnostic methods below belong solely to the evaluator.
    """
    schema = "worldzero-state-v2"
    plugin_schema = "worldzero-state-v3"
    _plugin_snapshot_fields = frozenset({
        "schema", "seed", "config", "law", "symbols", "home", "fertile",
        "resources", "modules", "time", "regime", "mechanism_enabled",
        "agent", "rng", "pending", "proposals", "events", "record",
        "event_count", "history_hash", "first_assembly", "assemblies",
        "conversions", "conversions_without_living_agent",
        "integrated_motif_time", "audit", "module_states", "family",
    })
    _plugin_family_fields = frozenset({
        "channels", "calibration_suite_sha256", "descriptor", "derived",
        "experimental", "fingerprint", "instance", "official", "origin", "proposal_records",
        "private_transition_records",
    })
    _plugin_instance_fields = frozenset({
        "enabled", "family_id", "family_version", "hidden_parameters",
        "private_state",
    })
    _plugin_agent_fields = frozenset({
        "position", "energy", "born", "generation", "age", "alive",
        "termination", "inventory", "decisions", "invalid_actions",
        "raw_consumed", "rich_consumed", "memory", "last_result",
    })
    _pcg64_fields = frozenset({"bit_generator", "state", "has_uint32", "uinteger"})
    _pcg64_state_fields = frozenset({"state", "inc"})
    _config_fields = frozenset(item.name for item in fields(Config))
    _plugin_law_fields = frozenset({"pair", "family", "geometry"})
    _audit_fields = frozenset({
        "initial_energy", "external_energy", "dissipated_energy",
        "initial_material", "incoming_material", "outgoing_material",
    })
    _event_kinds = frozenset({
        "initialized", "birth", "family_error", "death_drop", "death",
        "retired", "family_evidence", "physics", "decision", "assembly",
        "action", "intervention", "private_state_transition",
    })

    def __init__(self, seed: int, config: Config | None = None, law: Law | None = None,
                 *, family: RegisteredFamily | None = None, record: bool = True) -> None:
        if law is not None and family is not None:
            raise ValueError("Specify either legacy law or registered family, not both")
        if family is not None and not isinstance(family, RegisteredFamily):
            raise TypeError("family must be a RegisteredFamily")
        self.seed = int(seed)
        self.config = config or Config()
        c = self.config
        design = np.random.default_rng(derive_seed(seed, "layout-v2"))
        names = np.random.default_rng(derive_seed(seed, "symbols-v2"))
        self.rng = np.random.default_rng(derive_seed(seed, "noise-v2"))
        self._legacy_mode = family is None
        sample_draws: dict[str, Any] = {
            "module_count": 3,
            "raw_energy": c.raw_energy,
            "rich_energy": c.rich_energy,
            "dwell_duration": float(
                np.random.default_rng(derive_seed(seed, "law-parameters-v1")).uniform(2.0, 6.0)
            ),
        }
        if law is not None:
            sample_draws.update(
                compatibility_geometry=law.geometry,
                compatibility_pair=law.pair,
            )
        sample_context = SampleContext(
            sample_draws, named_seeds={"law": derive_seed(seed, "law-v2")},
        )
        if self._legacy_mode:
            from .laws.builtin import legacy_registered_family

            self._registered_family = legacy_registered_family(law.family if law is not None else "catalysis")
            self._family_instance = self._registered_family.family.sample(
                sample_context
            )
            self.law = law or self._compatibility_law(self._family_instance)
            self.schema = type(self).schema
        else:
            assert family is not None
            self._registered_family = family
            self._family_instance = family.family.sample(sample_context)
            self.law = self._compatibility_law(self._family_instance)
            self.schema = self.plugin_schema
        self._validate_family_instance()
        self._sampled_family_instance = self._family_instance
        self._family = self._registered_family.family
        self.symbols = [str(x) for x in names.permutation(["x0", "x1", "x2", "x3", "x4"])]
        self.home = (c.height // 2, c.width // 2)
        self.fertile = np.zeros((c.height, c.width), dtype=bool)
        for y in range(c.height):
            for x in range(c.width):
                self.fertile[y, x] = abs(y-self.home[0]) + abs(x-self.home[1]) <= 3
        # A second site varies by layout seed. Terrain is visible, its rates are not.
        cy, cx = int(design.integers(1, c.height-1)), int(design.integers(1, c.width-1))
        for y in range(c.height):
            for x in range(c.width):
                if abs(y-cy)+abs(x-cx) <= 2:
                    self.fertile[y, x] = True
        self._seed_symbols = tuple(self.symbols)
        self._seed_home = self.home
        self._seed_fertile = tuple(
            tuple(int(value) for value in row) for row in self.fertile
        )
        self.resources = np.where(self.fertile & (design.random(self.fertile.shape) < c.initial_resource_fraction), RAW, EMPTY).astype(np.int8)
        candidates = [(y,x) for y in range(c.height) for x in range(c.width)
                      if 1 <= abs(y-self.home[0])+abs(x-self.home[1]) <= 4]
        design.shuffle(candidates)
        positions: list[tuple[int,int]] = []
        for p in candidates:
            if all(abs(p[0]-q[0])+abs(p[1]-q[1]) >= 3 for q in positions):
                positions.append(p)
                if len(positions) == 3:
                    break
        if len(positions) != 3:
            raise RuntimeError("Generator cannot place three separated components")
        self.modules: list[tuple[int,int] | None] = positions
        self.time = 0.0
        self.regime = 0
        self.mechanism_enabled = self._family_instance.enabled
        self.agent: Agent | None = None
        self.record = record
        self.events: list[dict[str,Any]] = []
        self.history_hash = "0" * 64
        self.event_count = 0
        self.proposal_count = 0
        self._proposal_records: list[dict[str, Any]] = []
        self._private_transition_records: list[dict[str, Any]] = []
        self._pending: tuple[float,str,int,float] | None = None
        self._fertile_flat = [int(x) for x in np.flatnonzero(self.fertile)]
        self._family_channels = validate_channel_specs(
            self._registered_family.family.channels(self._family_instance, c)
        )
        self._family_channel_ids = frozenset(channel.channel_id for channel in self._family_channels)
        if self._family_channel_ids & {"source", "raw_decay", "rich_decay", "module_decay", "regime"}:
            raise ValueError("Family channel conflicts with a kernel-owned channel")
        self._channels = [("source", c.source_rate*len(self._fertile_flat)),
                          ("raw_decay", c.raw_decay*c.width*c.height),
                          ("rich_decay", c.rich_decay*c.width*c.height),
                          *((channel.channel_id, channel.envelope_rate) for channel in self._family_channels),
                          ("module_decay", c.module_decay*3),
                          ("regime", c.regime_rate)]
        self._bounds = np.cumsum([x[1] for x in self._channels]).tolist()
        self._total_rate = float(sum(x[1] for x in self._channels))
        self._field: set[int] = set()
        self._derived_family_state = DerivedLawState({}, False)
        self._module_states: list[dict[str, Any]] = [{}, {}, {}]
        self.first_assembly: float | None = None
        self.assemblies = 0
        self.conversions = 0
        self.conversions_without_living_agent = 0
        self.integrated_motif_time = 0.0
        self.audit = dict(initial_energy=self.resource_energy(), external_energy=0.0, dissipated_energy=0.0,
                          initial_material=self.material_count(), incoming_material=0, outgoing_material=0)
        self._update_field()
        self._log("initialized", home=list(self.home))
        self.spawn(generation=1)
        self._genesis_identity = {
            "mechanism_enabled": self.mechanism_enabled,
            "events": copy.deepcopy(self.events),
            "event_count": self.event_count,
            "history_hash": self.history_hash,
        }

    def _validate_family_instance(self) -> None:
        descriptor = self._registered_family.family.descriptor
        instance = self._family_instance
        if not isinstance(instance, FamilyInstance):
            raise TypeError("LawFamily.sample() must return a FamilyInstance")
        if (
            instance.family_id != descriptor.family_id
            or instance.family_version != descriptor.family_version
        ):
            raise ValueError("Sampled family instance identity does not match descriptor")

    def _validate_resulting_instance(self, instance: FamilyInstance) -> None:
        descriptor = self._family.descriptor
        if not isinstance(instance, FamilyInstance):
            self._transition_failure("intervention resulting instance is invalid")
        if instance.family_id != descriptor.family_id or instance.family_version != descriptor.family_version:
            self._transition_failure("intervention resulting instance identity does not match descriptor")

    def _validated_controls(self) -> ControlSuite:
        controls = self._call_family("controls", self._family_instance)
        if not isinstance(controls, ControlSuite):
            self._transition_failure("family controls callback returned an invalid suite")
        try:
            return ControlSuite(
                controls.null, controls.knockout, controls.broken, controls.retained,
            )
        except (TypeError, ValueError) as exc:
            self._transition_failure(f"family controls callback returned an invalid suite: {exc}")

    def _serialized_channels(self) -> list[dict[str, Any]]:
        return [
            {
                "channel_id": channel.channel_id,
                "draw_requirements": [item.value for item in channel.draw_requirements],
                "envelope_rate": channel.envelope_rate,
                "target_domain": channel.target_domain.value,
            }
            for channel in self._family_channels
        ]

    @staticmethod
    def _serialized_derived(value: DerivedLawState) -> dict[str, Any]:
        return {
            "affected_locations": list(value.affected_locations),
            "functional": value.functional,
            "state": thaw_json(value.state),
        }

    def _proposal_record(
        self,
        proposal: ProposalDraw,
        derived: DerivedLawState,
        transition: LawTransition | KernelProposalRejection | None,
    ) -> dict[str, Any] | None:
        if self._legacy_mode:
            return None
        if len(self._proposal_records) >= PLUGIN_PROPOSAL_RECORD_LIMIT:
            self._transition_failure("plugin proposal record limit exceeded")
        operations = [] if transition is None else [
            encode_transition_operation(operation) for operation in transition.operations
        ]
        accounting = {"energy_delta": 0.0, "material_delta": 0} if transition is None else {
            "energy_delta": transition.accounting.energy_delta,
            "material_delta": transition.accounting.material_delta,
        }
        evidence_events = [
            {
                "event_type": operation.event_type,
                "location_index": operation.location_index,
                "details": thaw_json(operation.details),
            }
            for operation in (() if transition is None else transition.operations)
            if isinstance(operation, EventEvidence)
        ]
        record = {
            "proposal": {
                "acceptance_uniform": proposal.acceptance_uniform,
                "channel_id": proposal.channel_id,
                "proposal_index": proposal.proposal_index,
                "simulated_time": proposal.simulated_time,
                "target_index": proposal.target_index,
            },
            "derived": self._serialized_derived(derived),
            "outcome": (
                "no_op" if transition is None else
                "rejected" if isinstance(transition, KernelProposalRejection) else "accepted"
            ),
            "operations": operations,
            "accounting": accounting,
            "declared_capabilities": [] if transition is None else sorted(transition.declared_capabilities),
            "evidence_events": evidence_events,
        }
        # Exercise the exact persistence validator before any caller commits the
        # semantic transition or evaluator history.
        try:
            self._validated_proposal_records(
                [record], proposal_count=self.proposal_count,
                simulated_time=self.time,
            )
        except (TypeError, ValueError) as exc:
            self._transition_failure(str(exc))
        return record

    def _append_proposal_record(
        self,
        proposal: ProposalDraw,
        derived: DerivedLawState,
        transition: LawTransition | KernelProposalRejection | None,
    ) -> None:
        record = self._proposal_record(proposal, derived, transition)
        if record is not None:
            self._proposal_records.append(record)

    def _validated_proposal_records(
        self, value: object, *, proposal_count: int, simulated_time: float,
    ) -> list[dict[str, Any]]:
        if type(value) is not list or len(value) > PLUGIN_PROPOSAL_RECORD_LIMIT:
            raise ValueError("Plugin snapshot proposal records are invalid")
        self._validate_exact_json(value, path="Plugin proposal records")
        result = copy.deepcopy(value)
        channel_specs = {
            channel.channel_id: channel for channel in self._family_channels
        }
        if "proposal_filter" in self._family.descriptor.capabilities:
            channel_specs["raw_decay"] = ChannelSpec(
                "raw_decay", self.config.raw_decay * self.config.width * self.config.height,
                TargetDomain.CELL,
                (DrawRequirement.TARGET_INDEX, DrawRequirement.ACCEPTANCE_UNIFORM),
            )
        descriptor_capabilities = self._family.descriptor.capabilities
        cell_count = self.config.width * self.config.height
        previous_index = 0
        previous_time = -math.inf
        for record in result:
            if type(record) is not dict or set(record) != {
                "proposal", "derived", "outcome", "operations", "accounting",
                "declared_capabilities", "evidence_events",
            }:
                raise ValueError("Plugin snapshot proposal record fields are invalid")
            proposal = record["proposal"]
            if type(proposal) is not dict or set(proposal) != {
                "acceptance_uniform", "channel_id", "proposal_index", "simulated_time", "target_index",
            }:
                raise ValueError("Plugin snapshot proposal draw is invalid")
            ProposalDraw(**proposal)
            channel = channel_specs.get(proposal["channel_id"])
            if channel is None:
                raise ValueError("Plugin snapshot proposal channel is not frozen")
            index = proposal["proposal_index"]
            proposal_time = proposal["simulated_time"]
            target = proposal["target_index"]
            acceptance = proposal["acceptance_uniform"]
            if (
                type(index) is not int or index <= previous_index
                or index > proposal_count
            ):
                raise ValueError("Plugin snapshot proposal index sequence is invalid")
            if (
                type(proposal_time) is not float
                or proposal_time < previous_time
                or proposal_time > simulated_time + 1e-9
            ):
                raise ValueError("Plugin snapshot proposal time sequence is invalid")
            if type(target) is not int:
                raise ValueError("Plugin snapshot proposal target type is invalid")
            if channel.target_domain is TargetDomain.CELL:
                valid_target = 0 <= target < cell_count
            elif channel.target_domain is TargetDomain.MODULE:
                valid_target = 0 <= target < 3
            else:
                valid_target = target == 0
            if not valid_target:
                raise ValueError(
                    "Plugin snapshot proposal target is out of bounds for its frozen domain"
                )
            if (
                DrawRequirement.TARGET_INDEX not in channel.draw_requirements
                and target != 0
            ):
                raise ValueError("Plugin snapshot proposal target draw default is invalid")
            if type(acceptance) is not float:
                raise ValueError("Plugin snapshot proposal acceptance type is invalid")
            if (
                DrawRequirement.ACCEPTANCE_UNIFORM not in channel.draw_requirements
                and acceptance != 0.0
            ):
                raise ValueError("Plugin snapshot proposal acceptance draw default is invalid")
            previous_index = index
            previous_time = proposal_time
            derived = record["derived"]
            if type(derived) is not dict or set(derived) != {
                "affected_locations", "functional", "state",
            }:
                raise ValueError("Plugin snapshot proposal derived state is invalid")
            if (
                type(derived["affected_locations"]) is not list
                or type(derived["functional"]) is not bool
                or type(derived["state"]) is not dict
            ):
                raise ValueError("Plugin snapshot proposal derived state types are invalid")
            derived_state = DerivedLawState(
                derived["state"], derived["functional"],
                tuple(derived["affected_locations"]),
            )
            if any(location >= cell_count for location in derived_state.affected_locations):
                raise ValueError("Plugin snapshot proposal derived target is out of bounds")
            if record["outcome"] not in {"accepted", "no_op", "rejected"}:
                raise ValueError("Plugin snapshot proposal outcome is invalid")
            if type(record["operations"]) is not list:
                raise ValueError("Plugin snapshot proposal operations are invalid")
            decoded_operations = tuple(
                decode_transition_operation(operation) for operation in record["operations"]
            )
            accounting = record["accounting"]
            if type(accounting) is not dict or set(accounting) != {"energy_delta", "material_delta"}:
                raise ValueError("Plugin snapshot proposal accounting is invalid")
            if (
                type(accounting["material_delta"]) is not int
                or type(accounting["energy_delta"]) is not float
            ):
                raise ValueError("Plugin snapshot proposal accounting types are invalid")
            AccountingDelta(accounting["material_delta"], accounting["energy_delta"])
            capabilities = record["declared_capabilities"]
            if (
                type(capabilities) is not list
                or any(type(item) is not str for item in capabilities)
                or capabilities != sorted(set(capabilities))
                or not set(capabilities).issubset(descriptor_capabilities)
            ):
                raise ValueError("Plugin snapshot proposal capabilities are invalid")
            if type(record["evidence_events"]) is not list:
                raise ValueError("Plugin snapshot proposal evidence is invalid")
            expected_evidence = [
                {
                    "event_type": operation.event_type,
                    "location_index": operation.location_index,
                    "details": thaw_json(operation.details),
                }
                for operation in decoded_operations
                if isinstance(operation, EventEvidence)
            ]
            if record["evidence_events"] != expected_evidence:
                raise ValueError("Plugin snapshot proposal evidence disagrees with operations")
            if record["outcome"] == "no_op" and not (
                not record["operations"]
                and accounting == {"energy_delta": 0.0, "material_delta": 0}
                and not record["declared_capabilities"]
            ):
                raise ValueError("Plugin snapshot no-op proposal record is inconsistent")
            if record["outcome"] == "accepted":
                LawTransition(
                    decoded_operations, AccountingDelta(
                        accounting["material_delta"], accounting["energy_delta"]
                    ), frozenset(capabilities),
                )
                material_delta = 0
                energy_delta = 0.0
                energies = {
                    EMPTY: 0.0,
                    RAW: self.config.raw_energy,
                    RICH: self.config.rich_energy,
                }
                for operation in decoded_operations:
                    if isinstance(operation, ResourceReplacement):
                        if operation.cell_index >= cell_count:
                            raise ValueError(
                                "Plugin snapshot proposal operation target is out of bounds"
                            )
                        if operation.expected_value not in energies:
                            raise ValueError(
                                "Plugin snapshot proposal operation expected value is invalid"
                            )
                        if operation.replacement_value not in energies:
                            raise ValueError(
                                "Plugin snapshot proposal operation replacement value is invalid"
                            )
                        material_delta += (
                            int(operation.replacement_value != EMPTY)
                            - int(operation.expected_value != EMPTY)
                        )
                        energy_delta += (
                            energies[operation.replacement_value]
                            - energies[operation.expected_value]
                        )
                    elif isinstance(operation, ResourcePreservation):
                        if operation.cell_index >= cell_count or operation.expected_value not in energies:
                            raise ValueError("Plugin snapshot proposal operation target is invalid")
                    elif isinstance(operation, ModulePositionChange):
                        for position in (
                            operation.expected_position, operation.replacement_position,
                        ):
                            if position is not None and not (
                                0 <= position[0] < self.config.height
                                and 0 <= position[1] < self.config.width
                            ):
                                raise ValueError("Plugin snapshot proposal operation target is invalid")
                        material_delta += (
                            int(operation.replacement_position is not None)
                            - int(operation.expected_position is not None)
                        )
                    elif isinstance(operation, EventEvidence):
                        if (
                            operation.location_index is not None
                            and operation.location_index >= cell_count
                        ):
                            raise ValueError("Plugin snapshot proposal operation target is invalid")
                if (
                    material_delta != accounting["material_delta"]
                    or energy_delta != accounting["energy_delta"]
                ):
                    raise ValueError("Plugin snapshot proposal operation accounting is invalid")
            if record["outcome"] == "rejected":
                try:
                    KernelProposalRejection(
                        ProposalDraw(**proposal), decoded_operations, frozenset(capabilities),
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("Plugin snapshot rejected proposal record is invalid") from exc
                if accounting != {"energy_delta": 0.0, "material_delta": 0}:
                    raise ValueError("Plugin snapshot rejected proposal accounting is invalid")
        return result

    def _compatibility_law(self, instance: FamilyInstance) -> Law | _PluginLawView:
        pair_value = instance.hidden_parameters.get("pair", (0, 1))
        geometry = str(instance.hidden_parameters.get("geometry", "adjacent"))
        try:
            pair = (int(pair_value[0]), int(pair_value[1]))  # type: ignore[index]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("Family instance pair cannot support the fixed substrate") from exc
        short_name = instance.family_id.partition(":")[2]
        try:
            return Law(pair, short_name, geometry)
        except ValueError:
            return _PluginLawView(pair, instance.family_id, geometry)

    def _sync_legacy_family(self) -> bool:
        if not self._legacy_mode:
            return False
        hidden_pair = tuple(self._family_instance.hidden_parameters.get("pair", ()))
        hidden_geometry = self._family_instance.hidden_parameters.get("geometry")
        expected_id = f"worldzero:{self.law.family}"
        if (
            self._family_instance.family_id == expected_id
            and hidden_pair == self.law.pair
            and hidden_geometry == self.law.geometry
            and self._family_instance.enabled == self.mechanism_enabled
        ):
            return False
        from .laws.builtin import legacy_registered_family

        self._registered_family = legacy_registered_family(self.law.family)
        self._family = self._registered_family.family
        sampled = self._family.sample(
            SampleContext(
                {
                    "compatibility_geometry": self.law.geometry,
                    "compatibility_pair": self.law.pair,
                    "module_count": 3,
                    "raw_energy": self.config.raw_energy,
                    "rich_energy": self.config.rich_energy,
                },
                named_seeds={"law": derive_seed(self.seed, "law-v2")},
            )
        )
        self._family_instance = FamilyInstance(
            sampled.family_id,
            sampled.family_version,
            sampled.hidden_parameters,
            sampled.private_state,
            enabled=self.mechanism_enabled,
        )
        return True

    def _public_view(self) -> PublicSubstrateView:
        agent_position = self.agent.position if self.agent is not None else self.home
        return PublicSubstrateView(
            width=self.config.width,
            height=self.config.height,
            agent_position=agent_position,
            module_positions=tuple(self.modules),
            module_states=tuple(copy.deepcopy(self._module_states)),
            resources=tuple(tuple(int(value) for value in row) for row in self.resources),
            terrain=tuple(tuple(int(value) for value in row) for row in self.fertile.astype(int)),
            simulated_time=self.time,
        )

    def _substrate_view(self) -> SubstrateView:
        public = self._public_view()
        return SubstrateView(
            width=public.width,
            height=public.height,
            agent_position=public.agent_position,
            module_positions=public.module_positions,
            module_states=public.module_states,
            resources=public.resources,
            terrain=public.terrain,
            simulated_time=public.simulated_time,
            kernel_counters={
                "assemblies": self.assemblies,
                "conversions": self.conversions,
                "proposal_count": self.proposal_count,
            },
        )

    def _candidate_substrate_view(
        self,
        resources: np.ndarray,
        modules: list[tuple[int, int] | None],
        module_states: list[dict[str, Any]],
    ) -> SubstrateView:
        agent_position = self.agent.position if self.agent is not None else self.home
        grid = resources.reshape(self.config.height, self.config.width)
        return SubstrateView(
            width=self.config.width,
            height=self.config.height,
            agent_position=agent_position,
            module_positions=tuple(modules),
            module_states=tuple(copy.deepcopy(module_states)),
            resources=tuple(tuple(int(value) for value in row) for row in grid),
            terrain=tuple(tuple(int(value) for value in row) for row in self.fertile.astype(int)),
            simulated_time=self.time,
            kernel_counters={
                "assemblies": self.assemblies,
                "conversions": self.conversions,
                "proposal_count": self.proposal_count,
            },
        )

    def _family_error(self, callback: str, exc: BaseException) -> None:
        self._log(
            "family_error",
            callback=callback,
            error_type=type(exc).__name__,
            message=str(exc)[:240],
        )

    def _call_family(self, callback: str, *args: Any) -> Any:
        try:
            return getattr(self._family, callback)(*args)
        except Exception as exc:
            self._family_error(callback, exc)
            raise FamilyCallbackError(
                f"Law family {self._family.descriptor.family_id} callback {callback} failed"
            ) from exc

    def _derive_family(self) -> DerivedLawState:
        self._synchronize_private_state(self._substrate_view())
        value = self._call_family("derive", self._substrate_view(), self._family_instance)
        if not isinstance(value, DerivedLawState):
            exc = TypeError("LawFamily.derive() must return DerivedLawState")
            self._family_error("derive", exc)
            raise FamilyCallbackError(str(exc))
        return value

    @staticmethod
    def _serialized_private_transition_view(view: SubstrateView) -> dict[str, Any]:
        return {
            "agent_position": list(view.agent_position),
            "height": view.height,
            "kernel_counters": dict(view.kernel_counters),
            "module_positions": [
                None if position is None else list(position)
                for position in view.module_positions
            ],
            "module_states": [thaw_json(state) for state in view.module_states],
            "resources": [list(row) for row in view.resources],
            "simulated_time": view.simulated_time,
            "terrain": [list(row) for row in view.terrain],
            "width": view.width,
        }

    @staticmethod
    def _private_transition_view(value: object) -> SubstrateView:
        if type(value) is not dict or set(value) != {
            "agent_position", "height", "kernel_counters", "module_positions",
            "module_states", "resources", "simulated_time", "terrain", "width",
        }:
            raise ValueError("Plugin snapshot private transition trigger fields are invalid")
        try:
            return SubstrateView(
                width=value["width"],
                height=value["height"],
                agent_position=value["agent_position"],
                module_positions=tuple(value["module_positions"]),
                module_states=tuple(value["module_states"]),
                resources=tuple(tuple(row) for row in value["resources"]),
                terrain=tuple(tuple(row) for row in value["terrain"]),
                simulated_time=value["simulated_time"],
                kernel_counters=value["kernel_counters"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Plugin snapshot private transition trigger is invalid") from exc

    def _private_state_transition_result(
        self, transition: PrivateStateTransition, instance: FamilyInstance,
        view: SubstrateView,
    ) -> tuple[FamilyInstance, dict[str, Any]]:
        if transition.expected_state != instance.private_state:
            self._transition_failure("private state transition expected state mismatch")
        if transition.declared_capabilities != frozenset({"private_state_transition"}):
            self._transition_failure("private state transition capabilities are invalid")
        if not transition.declared_capabilities.issubset(self._family.descriptor.capabilities):
            self._transition_failure("private state transition capability is undeclared")
        record = {
            "enabled": instance.enabled,
            "expected_state": thaw_json(instance.private_state),
            "family_id": instance.family_id,
            "family_version": instance.family_version,
            "fingerprint": self._registered_family.fingerprint,
            "proposal_index": self.proposal_count,
            "replacement_state": thaw_json(transition.replacement_state),
            "sequence": len(self._private_transition_records) + 1,
            "simulated_time": view.simulated_time,
            "trigger_view": self._serialized_private_transition_view(view),
        }
        record["transition_sha256"] = digest(record)
        return transition.resulting_instance(instance), record

    def _commit_private_transition_record(self, record: dict[str, Any]) -> None:
        """Commit one evaluator-owned private transition and its history binding."""

        if not self.record:
            raise FamilyTransitionError(
                "plugin private state transitions require record=True for auditability"
            )
        if len(self._private_transition_records) >= PLUGIN_PRIVATE_TRANSITION_RECORD_LIMIT:
            raise FamilyTransitionError("plugin private transition record limit exceeded")
        expected_sequence = len(self._private_transition_records) + 1
        if record.get("sequence") != expected_sequence:
            raise FamilyTransitionError("plugin private transition sequence is invalid")
        self._private_transition_records.append(copy.deepcopy(record))
        self._log(
            "private_state_transition",
            sequence=expected_sequence,
            transition_sha256=record["transition_sha256"],
        )

    def _synchronized_instance(
        self, view: SubstrateView, instance: FamilyInstance,
    ) -> tuple[FamilyInstance, dict[str, Any] | None]:
        transition = self._call_family("synchronize_private_state", view, instance)
        if transition is None:
            return instance, None
        if not isinstance(transition, PrivateStateTransition):
            self._transition_failure(
                "synchronize_private_state must return PrivateStateTransition or None"
            )
        return self._private_state_transition_result(transition, instance, view)

    def _synchronize_private_state(self, view: SubstrateView) -> None:
        result, record = self._synchronized_instance(view, self._family_instance)
        if record is not None:
            self._commit_private_transition_record(record)
        self._family_instance = result
        self.mechanism_enabled = result.enabled

    def _validated_private_transition_records(
        self,
        sampled: FamilyInstance,
        candidate: FamilyInstance,
        value: object,
        *,
        proposal_count: int,
        simulated_time: float,
        events: object,
        record_enabled: bool,
    ) -> list[dict[str, Any]]:
        if type(value) is not list or len(value) > PLUGIN_PRIVATE_TRANSITION_RECORD_LIMIT:
            raise ValueError("Plugin snapshot private transition records are invalid")
        self._validate_exact_json(value, path="Plugin private transition records")
        records = copy.deepcopy(value)
        if type(events) is not list:
            raise ValueError("Plugin snapshot private transition events are invalid")
        transition_events = [
            event for event in events
            if isinstance(event, Mapping)
            and event.get("kind") == "private_state_transition"
        ]
        if not record_enabled and (records or transition_events):
            raise ValueError(
                "Plugin snapshot record-false private transitions are unauditable"
            )
        if len(transition_events) != len(records):
            raise ValueError(
                "Plugin snapshot private transition event sequence is incomplete"
            )
        current = sampled
        previous_time = -math.inf
        previous_proposal = -1
        replay_enabled = sampled.enabled
        event_cursor = 0
        for sequence, (raw, transition_event) in enumerate(
            zip(records, transition_events), start=1,
        ):
            if not isinstance(raw, Mapping) or set(raw) != {
                "enabled", "expected_state", "family_id", "family_version", "fingerprint",
                "proposal_index", "replacement_state", "sequence", "simulated_time",
                "transition_sha256", "trigger_view",
            }:
                raise ValueError("Plugin snapshot private transition record fields are invalid")
            while event_cursor < len(events) and events[event_cursor] is not transition_event:
                event = events[event_cursor]
                if (
                    isinstance(event, Mapping)
                    and event.get("kind") == "intervention"
                    and event.get("intervention") in {
                        "matched_null", "mechanism_knockout",
                    }
                ):
                    replay_enabled = False
                event_cursor += 1
            if event_cursor >= len(events):
                raise ValueError("Plugin snapshot private transition event order is invalid")
            event_cursor += 1
            if (
                raw["family_id"] != sampled.family_id
                or raw["family_version"] != sampled.family_version
                or raw["fingerprint"] != self._registered_family.fingerprint
                or type(raw["enabled"]) is not bool
                or raw["enabled"] is not replay_enabled
                or raw["expected_state"] != thaw_json(current.private_state)
                or raw["sequence"] != sequence
            ):
                raise ValueError("Plugin snapshot private state transition identity is invalid")
            commitment = {
                key: copy.deepcopy(item)
                for key, item in raw.items()
                if key != "transition_sha256"
            }
            if (
                type(raw["transition_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", raw["transition_sha256"]) is None
                or digest(commitment) != raw["transition_sha256"]
            ):
                raise ValueError("Plugin snapshot private state transition digest is invalid")
            proposal_index = raw["proposal_index"]
            record_time = raw["simulated_time"]
            if (type(proposal_index) is not int or proposal_index < previous_proposal
                    or proposal_index > proposal_count):
                raise ValueError("Plugin snapshot private state proposal sequence is invalid")
            if (type(record_time) is not float or not math.isfinite(record_time)
                    or record_time < previous_time or record_time > simulated_time):
                raise ValueError("Plugin snapshot private state time sequence is invalid")
            trigger = self._private_transition_view(raw["trigger_view"])
            if (
                trigger.simulated_time != record_time
                or trigger.kernel_counters.get("proposal_count") != proposal_index
                or set(trigger.kernel_counters) != {
                    "assemblies", "conversions", "proposal_count",
                }
                or trigger.width != self.config.width
                or trigger.height != self.config.height
                or len(trigger.module_positions) != 3
                or any(
                    position is not None and not (
                        0 <= position[0] < trigger.height
                        and 0 <= position[1] < trigger.width
                    )
                    for position in trigger.module_positions
                )
                or len({
                    position for position in trigger.module_positions
                    if position is not None
                }) != sum(position is not None for position in trigger.module_positions)
                or not (0 <= trigger.agent_position[0] < trigger.height)
                or not (0 <= trigger.agent_position[1] < trigger.width)
                or any(value not in {EMPTY, RAW, RICH}
                       for row in trigger.resources for value in row)
                or any(value not in {0, 1}
                       for row in trigger.terrain for value in row)
            ):
                raise ValueError("Plugin snapshot private state trigger context is inconsistent")
            expected_event = {
                "kind": "private_state_transition",
                "proposal_index": proposal_index,
                "sequence": sequence,
                "time": record_time,
                "transition_sha256": raw["transition_sha256"],
            }
            if transition_event != expected_event:
                raise ValueError("Plugin snapshot private transition event binding is invalid")
            replay_instance = FamilyInstance(
                current.family_id, current.family_version, current.hidden_parameters,
                current.private_state, replay_enabled,
            )
            try:
                transition = self._family.synchronize_private_state(trigger, replay_instance)
            except Exception as exc:
                raise ValueError("Plugin snapshot private state callback replay failed") from exc
            if (
                not isinstance(transition, PrivateStateTransition)
                or transition.expected_state != replay_instance.private_state
                or thaw_json(transition.replacement_state) != raw["replacement_state"]
                or transition.declared_capabilities != frozenset({"private_state_transition"})
            ):
                raise ValueError("Plugin snapshot private state semantic transition is invalid")
            current = transition.resulting_instance(replay_instance)
            previous_time = record_time
            previous_proposal = proposal_index
        if current.private_state != candidate.private_state:
            raise ValueError("Plugin snapshot private state transition sequence does not reach final state")
        if not records and candidate.private_state != sampled.private_state:
            raise ValueError("Plugin snapshot private state transition records are missing")
        return records

    def neighbors(self, p: tuple[int,int]) -> list[tuple[int,int]]:
        y,x = p
        return [(y+dy,x+dx) for dy,dx in DIRECTIONS.values()
                if 0 <= y+dy < self.config.height and 0 <= x+dx < self.config.width]

    def _log(self, kind: str, **values: Any) -> None:
        row = {"time": self.time, "kind": kind, "proposal_index": self.proposal_count, **values}
        self.history_hash = digest([self.history_hash, row])
        self.event_count += 1
        if self.record:
            self.events.append(row)

    def clone(self, *, record: bool | None = None) -> World:
        try:
            cloned_family = copy.deepcopy(self._family)
        except Exception as exc:
            raise FamilyCallbackError("Law family executable state cannot be isolated") from exc
        if cloned_family is self._family:
            raise FamilyCallbackError("Law family executable state cannot be isolated")
        cloned_registration = RegisteredFamily(
            cloned_family,
            self._registered_family.origin,
            self._registered_family.official,
            self._registered_family.fingerprint,
        )
        memo = {
            id(self._registered_family): cloned_registration,
            id(self._family): cloned_family,
            id(self._family_instance): self._family_instance,
            id(self._sampled_family_instance): self._sampled_family_instance,
            id(self._derived_family_state): self._derived_family_state,
        }
        other = copy.deepcopy(self, memo)
        if record is not None:
            other.record = record
        return other

    def snapshot(self) -> dict[str,Any]:
        legacy = dict(schema=self.schema, seed=self.seed, config=asdict(self.config), law=asdict(self.law),
                    symbols=self.symbols, home=list(self.home), fertile=self.fertile.astype(int).tolist(),
                    resources=self.resources.tolist(), modules=[list(p) if p is not None else None for p in self.modules],
                    time=self.time, regime=self.regime, mechanism_enabled=self.mechanism_enabled,
                    agent=asdict(self.agent) if self.agent else None, rng=copy.deepcopy(self.rng.bit_generator.state),
                    pending=list(self._pending) if self._pending else None, proposals=self.proposal_count,
                    events=copy.deepcopy(self.events), record=self.record, event_count=self.event_count,
                    history_hash=self.history_hash, first_assembly=self.first_assembly, assemblies=self.assemblies,
                    conversions=self.conversions, conversions_without_living_agent=self.conversions_without_living_agent,
                    integrated_motif_time=self.integrated_motif_time, audit=copy.deepcopy(self.audit))
        if self._legacy_mode:
            return legacy
        legacy["law"]["pair"] = list(legacy["law"]["pair"])
        if legacy["agent"] is not None:
            legacy["agent"]["position"] = list(legacy["agent"]["position"])
        derived = self._derive_family()
        legacy["module_states"] = copy.deepcopy(self._module_states)
        legacy["family"] = {
            "channels": self._serialized_channels(),
            "calibration_suite_sha256": calibration_suite_fingerprint(self._family),
            "descriptor": self._family.descriptor.persistence_dict(),
            "derived": self._serialized_derived(derived),
            "experimental": self._registered_family.experimental,
            "fingerprint": self._registered_family.fingerprint,
            "instance": {
                "enabled": self._family_instance.enabled,
                "family_id": self._family_instance.family_id,
                "family_version": self._family_instance.family_version,
                "hidden_parameters": thaw_json(self._family_instance.hidden_parameters),
                "private_state": thaw_json(self._family_instance.private_state),
            },
            "official": self._registered_family.official,
            "origin": self._registered_family.origin,
            "private_transition_records": copy.deepcopy(
                self._private_transition_records
            ),
            "proposal_records": copy.deepcopy(self._proposal_records),
        }
        return legacy

    @staticmethod
    def _state_v3_finite(name: str, value: object, *, nonnegative: bool = False) -> float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Plugin snapshot state-v3 {name} must be finite")
        result = float(value)
        if nonnegative and result < 0.0:
            raise ValueError(f"Plugin snapshot state-v3 {name} must be nonnegative")
        return result

    @classmethod
    def _validate_exact_json(cls, value: object, *, path: str = "Plugin snapshot state-v3") -> None:
        """Reject Python coercions: state-v3 and trace-v4 persist exact JSON values."""

        if value is None or type(value) in {bool, int, str}:
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError(f"{path} contains a non-finite JSON number")
            return
        if type(value) is list:
            for index, item in enumerate(value):
                cls._validate_exact_json(item, path=f"{path}[{index}]")
            return
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"{path} contains a non-string JSON object key")
                cls._validate_exact_json(item, path=f"{path}.{key}")
            return
        raise TypeError(f"{path} contains a non-JSON value")

    @classmethod
    def _validated_state_v3_config(cls, value: object) -> Config:
        if type(value) is not dict or set(value) != cls._config_fields:
            raise ValueError("Plugin snapshot config fields are invalid")
        try:
            return Config(**value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Plugin snapshot config is invalid") from exc

    @classmethod
    def _validated_state_v3_law(cls, value: object) -> dict[str, Any]:
        if type(value) is not dict or set(value) != cls._plugin_law_fields:
            raise ValueError("Plugin snapshot compatibility law fields are invalid")
        pair = value["pair"]
        if (type(pair) is not list or len(pair) != 2
                or any(type(item) is not int or item not in range(3) for item in pair)
                or len(set(pair)) != 2):
            raise ValueError("Plugin snapshot compatibility law pair is invalid")
        if type(value["family"]) is not str or not value["family"]:
            raise ValueError("Plugin snapshot compatibility law family is invalid")
        if type(value["geometry"]) is not str or not value["geometry"]:
            raise ValueError("Plugin snapshot compatibility law geometry is invalid")
        return copy.deepcopy(value)

    @classmethod
    def _validated_state_v3_position(
        cls, name: str, value: object, *, config: Config,
    ) -> tuple[int, int]:
        if (type(value) is not list or len(value) != 2
                or any(type(item) is not int for item in value)):
            raise ValueError(f"Plugin snapshot {name} position is invalid")
        position = (value[0], value[1])
        if not (0 <= position[0] < config.height and 0 <= position[1] < config.width):
            raise ValueError(f"Plugin snapshot {name} position is out of bounds")
        return position

    @classmethod
    def _validated_state_v3_grid(
        cls, name: str, value: object, *, config: Config,
        vocabulary: frozenset[int],
    ) -> list[list[int]]:
        if type(value) is not list or len(value) != config.height:
            raise ValueError(f"Plugin snapshot {name} matrix dimensions are invalid")
        result: list[list[int]] = []
        for row in value:
            if type(row) is not list or len(row) != config.width:
                raise ValueError(f"Plugin snapshot {name} matrix dimensions are invalid")
            if any(type(item) is not int or item not in vocabulary for item in row):
                raise ValueError(f"Plugin snapshot {name} matrix vocabulary is invalid")
            result.append(list(row))
        return result

    @classmethod
    def _validated_state_v3_audit(cls, value: object) -> dict[str, int | float]:
        if type(value) is not dict or set(value) != cls._audit_fields:
            raise ValueError("Plugin snapshot audit fields are invalid")
        result: dict[str, int | float] = {}
        for name in ("initial_energy", "external_energy", "dissipated_energy"):
            cls._state_v3_finite(
                f"audit {name}", value[name], nonnegative=True,
            )
            result[name] = value[name]
        for name in ("initial_material", "incoming_material", "outgoing_material"):
            item = value[name]
            if type(item) is not int or item < 0:
                raise ValueError(f"Plugin snapshot audit {name} is invalid")
            result[name] = item
        return result

    @classmethod
    def _validate_state_v3_events(
        cls, snapshot: dict[str, Any], *, expected_genesis: dict[str, Any],
    ) -> None:
        history = snapshot["history_hash"]
        if type(history) is not str or re.fullmatch(r"[0-9a-f]{64}", history) is None:
            raise ValueError("Plugin snapshot history hash is invalid")
        events = snapshot["events"]
        if type(events) is not list:
            raise ValueError("Plugin snapshot events must be a JSON array")
        if not snapshot["record"]:
            if events:
                raise ValueError("Plugin snapshot record-false events must be empty")
        else:
            if snapshot["event_count"] != len(events):
                raise ValueError("Plugin snapshot event count disagrees with recorded events")
            previous_time = -math.inf
            previous_proposal = -1
            recomputed = "0" * 64
            for event in events:
                if type(event) is not dict or not {"time", "kind", "proposal_index"}.issubset(event):
                    raise ValueError("Plugin snapshot event base fields are invalid")
                event_time = event["time"]
                if (type(event_time) not in (int, float) or not math.isfinite(event_time)
                        or event_time < 0.0 or event_time > snapshot["time"] + 1e-9
                        or event_time < previous_time):
                    raise ValueError("Plugin snapshot event time is invalid")
                kind = event["kind"]
                if type(kind) is not str or kind not in cls._event_kinds:
                    raise ValueError("Plugin snapshot event kind is invalid")
                proposal = event["proposal_index"]
                if (type(proposal) is not int or proposal < 0
                        or proposal > snapshot["proposals"] or proposal < previous_proposal):
                    raise ValueError("Plugin snapshot event proposal index is invalid")
                previous_time = float(event_time)
                previous_proposal = proposal
                recomputed = digest([recomputed, event])
            if recomputed != history:
                raise ValueError("Plugin snapshot history chain is invalid")
        if snapshot["time"] == 0.0:
            agent = snapshot["agent"]
            fresh = (
                type(agent) is dict and agent.get("alive") is True
                and agent.get("generation") == 1 and agent.get("born") == 0.0
                and agent.get("age") == 0.0 and agent.get("decisions") == 0
                and snapshot["mechanism_enabled"] is expected_genesis["mechanism_enabled"]
                and not snapshot["family"]["proposal_records"]
            )
            if fresh and (
                snapshot["proposals"] != 0 or any(
                    snapshot[name] != 0 for name in (
                        "assemblies", "conversions", "conversions_without_living_agent",
                    )
                ) or snapshot["integrated_motif_time"] != 0.0
            ):
                raise ValueError("Plugin snapshot time-zero counters are invalid")
            if fresh and (
                snapshot["event_count"] != expected_genesis["event_count"]
                or snapshot["history_hash"] != expected_genesis["history_hash"]
                or snapshot["events"] != expected_genesis["events"]
            ):
                raise ValueError("Plugin snapshot genesis event/history is invalid")

    def validate_plugin_trace_origin(self) -> None:
        """Validate an in-memory plugin world before it becomes trace evidence."""

        if self._legacy_mode:
            raise ValueError("Plugin trace-v4 origin requires a registered family")
        snapshot = self.snapshot()
        self._validate_exact_json(snapshot, path="Plugin trace-v4 origin")
        self._validate_state_v3_scalars(snapshot)
        self._validated_state_v3_config(snapshot["config"])
        law = self._validated_state_v3_law(snapshot["law"])
        expected_law = asdict(self._compatibility_law(self._sampled_family_instance))
        expected_law["pair"] = list(expected_law["pair"])
        if law != expected_law:
            raise ValueError("Plugin trace-v4 origin compatibility law is immutable")
        if (
            tuple(snapshot["symbols"]) != self._seed_symbols
            or tuple(snapshot["home"]) != self._seed_home
            or tuple(tuple(row) for row in snapshot["fertile"]) != self._seed_fertile
        ):
            raise ValueError("Plugin trace-v4 origin seed-derived layout is immutable")
        if (self._family_instance.hidden_parameters
                != self._sampled_family_instance.hidden_parameters):
            raise ValueError("Plugin trace-v4 origin sampled family state is immutable")
        validated_private_records = self._validated_private_transition_records(
            self._sampled_family_instance,
            self._family_instance,
            snapshot["family"]["private_transition_records"],
            proposal_count=self.proposal_count,
            simulated_time=self.time,
            events=snapshot["events"],
            record_enabled=self.record,
        )
        if validated_private_records != self._private_transition_records:
            raise ValueError("Plugin trace-v4 origin private transition records drift")
        self._validated_state_v3_grid(
            "fertile", snapshot["fertile"], config=self.config,
            vocabulary=frozenset({0, 1}),
        )
        self._validated_state_v3_grid(
            "resources", snapshot["resources"], config=self.config,
            vocabulary=frozenset({EMPTY, RAW, RICH}),
        )
        self._validated_state_v3_agent(
            snapshot["agent"], config=self.config, modules=self.modules,
            simulated_time=float(self.time),
        )
        self._validated_pcg64_state(snapshot["rng"])
        self._validated_state_v3_pending(snapshot["pending"])
        self._validated_proposal_records(
            snapshot["family"]["proposal_records"],
            proposal_count=snapshot["proposals"],
            simulated_time=snapshot["time"],
        )
        self._validated_state_v3_audit(snapshot["audit"])
        self._validate_state_v3_events(
            snapshot, expected_genesis=self._genesis_identity,
        )
        self._validate_plugin_fresh_observer(snapshot)
        error = self.accounting_error()
        if abs(error["energy"]) > 1e-7 or error["material"] != 0:
            raise ValueError("Plugin trace-v4 origin conservation ledger is invalid")

    @staticmethod
    def _validate_plugin_fresh_observer(snapshot: dict[str, Any]) -> None:
        """Require capture to begin at one observer's auditable birth boundary."""

        agent = snapshot["agent"]
        if type(agent) is not dict or agent["alive"] is not True:
            raise ValueError("Plugin trace-v4 origin requires a fresh living observer")
        if (
            not math.isclose(agent["age"], 0.0, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(
                agent["born"], snapshot["time"], rel_tol=0.0, abs_tol=1e-9,
            )
            or agent["decisions"] != 0
            or agent["invalid_actions"] != 0
            or agent["raw_consumed"] != 0
            or agent["rich_consumed"] != 0
            or agent["memory"] != ""
            or agent["last_result"] != {}
            or agent["termination"] is not None
            or agent["inventory"] is not None
        ):
            raise ValueError("Plugin trace-v4 origin observer is not semantically fresh")
        current_generation = [
            (index, event)
            for index, event in enumerate(snapshot["events"])
            if event.get("generation") == agent["generation"]
        ]
        if not current_generation:
            raise ValueError("Plugin trace-v4 origin has no matching birth event")
        birth_index, birth = current_generation[-1]
        if (
            birth.get("kind") != "birth"
            or not math.isclose(
                birth.get("time", math.inf), snapshot["time"],
                rel_tol=0.0, abs_tol=1e-9,
            )
            or birth.get("position") != agent["position"]
            or type(birth.get("energy")) not in (int, float)
            or not math.isclose(
                birth["energy"], agent["energy"], rel_tol=0.0, abs_tol=1e-9,
            )
        ):
            raise ValueError("Plugin trace-v4 origin birth does not match the observer")
        if any(
            event.get("kind") in {"action", "decision"}
            for event in snapshot["events"][birth_index + 1:]
        ):
            raise ValueError("Plugin trace-v4 origin contains policy activity after birth")

    @classmethod
    def _validate_state_v3_scalars(cls, snapshot: dict[str, Any]) -> None:
        if type(snapshot["seed"]) is not int or snapshot["seed"] < 0:
            raise ValueError("Plugin snapshot state-v3 seed is invalid")
        if type(snapshot["record"]) is not bool:
            raise ValueError("Plugin snapshot state-v3 record flag is invalid")
        now = cls._state_v3_finite("time", snapshot["time"], nonnegative=True)
        if type(snapshot["regime"]) is not int or snapshot["regime"] not in {0, 1}:
            raise ValueError("Plugin snapshot state-v3 regime is invalid")
        counters = (
            "proposals", "event_count", "assemblies", "conversions",
            "conversions_without_living_agent",
        )
        for field_name in counters:
            value = snapshot[field_name]
            if type(value) is not int or value < 0:
                raise ValueError(f"Plugin snapshot state-v3 {field_name} is invalid")
        if snapshot["conversions_without_living_agent"] > snapshot["conversions"]:
            raise ValueError("Plugin snapshot state-v3 conversion counters are inconsistent")
        integrated = cls._state_v3_finite(
            "integrated_motif_time", snapshot["integrated_motif_time"],
            nonnegative=True,
        )
        if integrated > now + 1e-9:
            raise ValueError("Plugin snapshot state-v3 integrated motif time exceeds time")
        first = snapshot["first_assembly"]
        assemblies = snapshot["assemblies"]
        if assemblies == 0:
            if first is not None:
                raise ValueError("Plugin snapshot state-v3 first assembly is inconsistent")
        else:
            first_value = cls._state_v3_finite(
                "first_assembly", first, nonnegative=True,
            )
            if first_value > now + 1e-9:
                raise ValueError("Plugin snapshot state-v3 first assembly exceeds time")

    @classmethod
    def _validated_state_v3_agent(
        cls, value: object, *, config: Config, modules: list[tuple[int, int] | None],
        simulated_time: float,
    ) -> Agent | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != cls._plugin_agent_fields:
            raise ValueError("Plugin snapshot agent fields are invalid")
        position = value["position"]
        if (not isinstance(position, list) or len(position) != 2
                or any(type(item) is not int for item in position)):
            raise ValueError("Plugin snapshot agent position is invalid")
        if not (0 <= position[0] < config.height and 0 <= position[1] < config.width):
            raise ValueError("Plugin snapshot agent position is out of bounds")
        energy = cls._state_v3_finite("agent energy", value["energy"], nonnegative=True)
        born = cls._state_v3_finite("agent born", value["born"], nonnegative=True)
        age = cls._state_v3_finite("agent age", value["age"], nonnegative=True)
        generation = value["generation"]
        if type(generation) is not int or generation <= 0:
            raise ValueError("Plugin snapshot agent generation is invalid")
        alive = value["alive"]
        if type(alive) is not bool:
            raise ValueError("Plugin snapshot agent alive flag is invalid")
        termination = value["termination"]
        if termination is not None and type(termination) is not str:
            raise ValueError("Plugin snapshot agent termination is invalid")
        inventory = value["inventory"]
        if inventory is not None and (
            type(inventory) is not int or inventory not in range(3)
        ):
            raise ValueError("Plugin snapshot agent inventory is invalid")
        if inventory is not None and modules[inventory] is not None:
            raise ValueError("Plugin snapshot agent inventory conflicts with module positions")
        decisions = value["decisions"]
        invalid_actions = value["invalid_actions"]
        if (type(decisions) is not int or decisions < 0
                or decisions > config.max_decisions):
            raise ValueError("Plugin snapshot agent decisions are invalid")
        if (type(invalid_actions) is not int or invalid_actions < 0
                or invalid_actions > decisions):
            raise ValueError("Plugin snapshot agent invalid-action count is invalid")
        for field_name in ("raw_consumed", "rich_consumed"):
            count = value[field_name]
            if type(count) is not int or count < 0:
                raise ValueError(f"Plugin snapshot agent {field_name} is invalid")
        memory = value["memory"]
        if type(memory) is not str or len(memory) > config.private_memory_chars:
            raise ValueError("Plugin snapshot agent memory is invalid")
        last_result = value["last_result"]
        if not isinstance(last_result, dict):
            raise ValueError("Plugin snapshot agent last_result is invalid")
        try:
            freeze_json(last_result, path="agent.last_result")
        except (TypeError, ValueError) as exc:
            raise ValueError("Plugin snapshot agent last_result is invalid") from exc
        if born > simulated_time + 1e-9 or born + age > simulated_time + 1e-9:
            raise ValueError("Plugin snapshot agent born/age/time causality is invalid")
        if alive:
            if termination is not None:
                raise ValueError("Plugin snapshot live agent termination must be null")
            if energy <= 0.0:
                raise ValueError("Plugin snapshot live agent energy must be positive")
            if age >= config.lifespan:
                raise ValueError("Plugin snapshot live agent age reached the lifespan boundary")
            if not math.isclose(born + age, simulated_time, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("Plugin snapshot live agent age does not match simulated time")
        else:
            if not isinstance(termination, str) or not termination:
                raise ValueError("Plugin snapshot dead agent termination is invalid")
            if memory != "" or last_result != {} or inventory is not None:
                raise ValueError("Plugin snapshot dead agent state was not cleared")
        detached = copy.deepcopy(value)
        detached["position"] = tuple(position)
        return Agent(**detached)

    @classmethod
    def _validated_pcg64_state(cls, value: object) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != cls._pcg64_fields:
            raise ValueError("Plugin snapshot RNG fields are invalid")
        nested = value["state"]
        if not isinstance(nested, dict) or set(nested) != cls._pcg64_state_fields:
            raise ValueError("Plugin snapshot RNG nested state fields are invalid")
        if value["bit_generator"] != "PCG64":
            raise ValueError("Plugin snapshot RNG identity is invalid")
        state = nested["state"]
        increment = nested["inc"]
        if type(state) is not int or not 0 <= state < 2 ** 128:
            raise ValueError("Plugin snapshot RNG state is invalid")
        if (type(increment) is not int or not 0 <= increment < 2 ** 128
                or increment % 2 != 1):
            raise ValueError("Plugin snapshot RNG increment is invalid")
        has_uint32 = value["has_uint32"]
        if type(has_uint32) is not int or has_uint32 not in {0, 1}:
            raise ValueError("Plugin snapshot RNG has_uint32 is invalid")
        uinteger = value["uinteger"]
        if type(uinteger) is not int or not 0 <= uinteger < 2 ** 32:
            raise ValueError("Plugin snapshot RNG uinteger is invalid")
        return copy.deepcopy(value)

    def _validated_state_v3_pending(self, value: object) -> tuple[float, str, int, float] | None:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("Plugin snapshot pending proposal fields are invalid")
        scheduled, channel, target, acceptance = value
        if type(scheduled) not in (int, float) or not math.isfinite(scheduled):
            raise ValueError("Plugin snapshot pending scheduled time is invalid")
        if scheduled < self.time:
            raise ValueError("Plugin snapshot pending proposal precedes simulated time")
        if type(channel) is not str:
            raise ValueError("Plugin snapshot pending channel is invalid")
        channel_rates = dict(self._channels)
        if channel not in channel_rates or channel_rates[channel] <= 0.0:
            raise ValueError("Plugin snapshot pending channel is not a live frozen channel")
        if type(target) is not int:
            raise ValueError("Plugin snapshot pending target is invalid")
        cell_count = self.config.width * self.config.height
        if channel == "source":
            valid_target = target in self._fertile_flat
        elif channel in {"raw_decay", "rich_decay"}:
            valid_target = 0 <= target < cell_count
        elif channel == "module_decay":
            valid_target = 0 <= target < 3
        elif channel == "regime":
            valid_target = target == 0
        else:
            specification = next(
                spec for spec in self._family_channels if spec.channel_id == channel
            )
            if specification.target_domain is TargetDomain.CELL:
                valid_target = 0 <= target < cell_count
            elif specification.target_domain is TargetDomain.MODULE:
                valid_target = 0 <= target < 3
            else:
                valid_target = target == 0
        if not valid_target:
            raise ValueError("Plugin snapshot pending target is outside its channel domain")
        if (type(acceptance) not in (int, float) or not math.isfinite(acceptance)
                or not 0.0 <= acceptance < 1.0):
            raise ValueError("Plugin snapshot pending acceptance uniform is invalid")
        return (scheduled, channel, target, acceptance)

    @classmethod
    def from_snapshot(
        cls, s: dict[str, Any], *, registry: Any | None = None,
        expected_sha256: str | None = None,
    ) -> World:
        """Restore a consistent snapshot, optionally authenticated by an external digest.

        Without ``expected_sha256`` this validates only the artifact's internal
        schema and deterministic relationships. A coherently rewritten portable
        artifact cannot authenticate its own historical origin.
        """
        if not isinstance(s, dict):
            raise ValueError("Snapshot must be an object")
        require_expected_sha256("snapshot", s, expected_sha256)
        schema = s.get("schema")
        if schema not in {cls.schema, cls.plugin_schema}:
            raise ValueError("Unsupported snapshot schema")
        if schema == cls.schema:
            law = dict(s["law"]); law["pair"] = tuple(law["pair"])
            w = cls(s["seed"], Config(**s["config"]), Law(**law), record=s["record"])
            candidate_instance = None
            expected_genesis = None
        else:
            if type(s) is not dict or set(s) != cls._plugin_snapshot_fields:
                raise ValueError("Plugin snapshot top-level fields are invalid")
            cls._validate_exact_json(s)
            if s["schema"] != cls.plugin_schema:
                raise ValueError("Plugin snapshot schema identity is invalid")
            cls._validate_state_v3_scalars(s)
            config = cls._validated_state_v3_config(s["config"])
            persisted_law = cls._validated_state_v3_law(s["law"])
            if type(s["mechanism_enabled"]) is not bool:
                raise ValueError("Plugin snapshot mechanism flag is invalid")
            family_data = s.get("family")
            if type(family_data) is not dict or set(family_data) != cls._plugin_family_fields:
                raise ValueError("Plugin snapshot family fields are invalid")
            descriptor = family_data.get("descriptor")
            if type(descriptor) is not dict or type(descriptor.get("family_id")) is not str:
                raise ValueError("Plugin snapshot family descriptor is invalid")
            registered = resolve_family(descriptor["family_id"], registry=registry)
            if registered.fingerprint != family_data.get("fingerprint"):
                raise ValueError("Plugin snapshot family fingerprint drift")
            if registered.family.descriptor.persistence_dict() != descriptor:
                raise ValueError("Plugin snapshot family descriptor drift")
            if calibration_suite_fingerprint(registered.family) != family_data.get(
                "calibration_suite_sha256"
            ):
                raise ValueError("Plugin snapshot family calibration suite drift")
            if family_data.get("origin") != registered.origin:
                raise ValueError("Plugin snapshot family origin drift")
            if family_data.get("official") is not registered.official:
                raise ValueError("Plugin snapshot family official status drift")
            if family_data.get("experimental") is not registered.experimental:
                raise ValueError("Plugin snapshot family experimental status drift")
            w = cls(
                s["seed"],
                config,
                family=registered,
                record=s["record"],
            )
            expected_genesis = w.snapshot()
            sampled_instance = w._family_instance
            instance_data = family_data.get("instance")
            if (type(instance_data) is not dict
                    or set(instance_data) != cls._plugin_instance_fields):
                raise ValueError("Plugin snapshot family instance fields are invalid")
            candidate_instance = FamilyInstance(
                family_id=instance_data["family_id"],
                family_version=instance_data["family_version"],
                hidden_parameters=instance_data["hidden_parameters"],
                private_state=instance_data["private_state"],
                enabled=instance_data["enabled"],
            )
            if (
                candidate_instance.family_id != registered.family.descriptor.family_id
                or candidate_instance.family_version != registered.family.descriptor.family_version
            ):
                raise ValueError("Plugin snapshot family instance identity is invalid")
            if candidate_instance.hidden_parameters != sampled_instance.hidden_parameters:
                raise ValueError("Plugin snapshot sampled hidden parameters are immutable")
            w.proposal_count = s["proposals"]
            w.time = s["time"]
            w._private_transition_records = w._validated_private_transition_records(
                w._sampled_family_instance,
                candidate_instance,
                family_data.get("private_transition_records"),
                proposal_count=s["proposals"],
                simulated_time=s["time"],
                events=s["events"],
                record_enabled=s["record"],
            )
            expected_law = asdict(w._compatibility_law(sampled_instance))
            expected_law["pair"] = list(expected_law["pair"])
            if persisted_law != expected_law:
                raise ValueError("Plugin snapshot compatibility law drift")
            if s["mechanism_enabled"] is not candidate_instance.enabled:
                raise ValueError("Plugin snapshot mechanism state disagrees with family instance")
            w._proposal_records = w._validated_proposal_records(
                family_data.get("proposal_records"),
                proposal_count=s["proposals"], simulated_time=s["time"],
            )

        if schema == cls.plugin_schema:
            assert expected_genesis is not None
            symbols = s["symbols"]
            if (type(symbols) is not list or len(symbols) != 5
                    or any(type(item) is not str for item in symbols)
                    or len(set(symbols)) != 5):
                raise ValueError("Plugin snapshot symbols are invalid")
            if symbols != expected_genesis["symbols"]:
                raise ValueError("Plugin snapshot seed-derived symbols drift")
            home = cls._validated_state_v3_position("home", s["home"], config=w.config)
            if list(home) != expected_genesis["home"]:
                raise ValueError("Plugin snapshot seed-derived home drift")
            fertile_values = cls._validated_state_v3_grid(
                "fertile", s["fertile"], config=w.config,
                vocabulary=frozenset({0, 1}),
            )
            if fertile_values != expected_genesis["fertile"]:
                raise ValueError("Plugin snapshot seed-derived fertile terrain drift")
            resource_values = cls._validated_state_v3_grid(
                "resources", s["resources"], config=w.config,
                vocabulary=frozenset({EMPTY, RAW, RICH}),
            )
            w.symbols = copy.deepcopy(symbols)
            w.home = home
            w.fertile = np.array(fertile_values, dtype=bool)
            w.resources = np.array(resource_values, dtype=np.int8)
        else:
            w.symbols = list(s["symbols"]); w.home = tuple(s["home"])
            w.fertile = np.array(s["fertile"], dtype=bool)
            w.resources = np.array(s["resources"], dtype=np.int8)
        raw_modules = s["modules"]
        if not isinstance(raw_modules, list) or len(raw_modules) != 3:
            raise ValueError("Snapshot module positions are invalid")
        modules: list[tuple[int, int] | None] = []
        for position in raw_modules:
            if position is None:
                modules.append(None)
                continue
            accepted_position_types = (list,) if schema == cls.plugin_schema else (list, tuple)
            if (not isinstance(position, accepted_position_types) or len(position) != 2
                    or any(type(item) is not int for item in position)):
                raise ValueError("Snapshot module position is invalid")
            candidate = (position[0], position[1])
            if not (0 <= candidate[0] < w.config.height and 0 <= candidate[1] < w.config.width):
                raise ValueError("Snapshot module position is out of bounds")
            modules.append(candidate)
        if len({position for position in modules if position is not None}) != sum(position is not None for position in modules):
            raise ValueError("Snapshot module positions overlap")
        w.modules = modules
        w.time = s["time"]; w.regime = s["regime"]; w.mechanism_enabled = s["mechanism_enabled"]
        if schema == cls.plugin_schema:
            module_states = s.get("module_states")
            if (type(module_states) is not list or len(module_states) != 3
                    or any(type(item) is not dict for item in module_states)):
                raise ValueError("Plugin snapshot module states are invalid")
            frozen_states = freeze_json(module_states, path="module_states")
            if not isinstance(frozen_states, tuple) or any(not isinstance(item, Mapping) for item in frozen_states):
                raise ValueError("Plugin snapshot module states are invalid")
            w._module_states = [thaw_json(item) for item in frozen_states]  # type: ignore[list-item]
        if schema == cls.plugin_schema:
            w.agent = cls._validated_state_v3_agent(
                s["agent"], config=w.config, modules=w.modules,
                simulated_time=float(w.time),
            )
            assert candidate_instance is not None
            sampled_instance = w._family_instance
            if candidate_instance.enabled != sampled_instance.enabled:
                if not sampled_instance.enabled or candidate_instance.enabled:
                    raise ValueError("Plugin snapshot enabled-state control provenance is invalid")
                try:
                    enabled_candidate = FamilyInstance(
                        candidate_instance.family_id, candidate_instance.family_version,
                        candidate_instance.hidden_parameters, candidate_instance.private_state,
                        True,
                    )
                    controls = w._family.controls(enabled_candidate)
                    validated_controls = ControlSuite(
                        controls.null, controls.knockout, controls.broken, controls.retained,
                    )
                    if (
                        validated_controls.knockout.kind is not ControlKind.KNOCKOUT
                        or validated_controls.null.kind is not ControlKind.NULL
                    ):
                        raise ValueError("declared disabled control is invalid")
                    explained = False
                    for disabled_control in (ControlKind.KNOCKOUT, ControlKind.NULL):
                        transition = w._family.intervene(
                            disabled_control, w._substrate_view(), enabled_candidate,
                        )
                        explained = explained or (
                            isinstance(transition, InterventionTransition)
                            and transition.control is disabled_control
                            and transition.resulting_instance == candidate_instance
                            and not transition.operations
                            and not transition.declared_capabilities
                            and transition.accounting == AccountingDelta()
                        )
                    if (
                        not explained
                        or validate_channel_specs(
                            w._family.channels(candidate_instance, w.config)
                        ) != w._family_channels
                    ):
                        raise ValueError("declared disabled control does not explain restored state")
                except Exception as exc:
                    raise ValueError(
                        "Plugin snapshot enabled-state control provenance is invalid"
                    ) from exc
            elif (candidate_instance.hidden_parameters != sampled_instance.hidden_parameters
                  or candidate_instance.enabled != sampled_instance.enabled):
                raise ValueError("Plugin snapshot sampled family instance is immutable")
            w._family_instance = candidate_instance
            w.law = w._compatibility_law(candidate_instance)
            rng_state = cls._validated_pcg64_state(s["rng"])
        else:
            a = s["agent"]
            if a is not None:
                a = dict(a); a["position"] = tuple(a["position"])
            w.agent = Agent(**a) if a else None
            rng_state = copy.deepcopy(s["rng"])
        w.rng.bit_generator.state = rng_state
        # Rebuild source-domain metadata before validating a pending source draw.
        w._fertile_flat = [int(x) for x in np.flatnonzero(w.fertile)]
        w._channels[0] = ("source", w.config.source_rate*len(w._fertile_flat))
        if schema == cls.plugin_schema:
            w._pending = w._validated_state_v3_pending(s["pending"])
        else:
            w._pending = tuple(s["pending"]) if s["pending"] else None
        w.proposal_count = s["proposals"]
        if schema == cls.plugin_schema:
            if s["conversions"] > s["proposals"]:
                raise ValueError("Plugin snapshot conversion counter exceeds proposals")
            assert expected_genesis is not None
            cls._validate_state_v3_events(s, expected_genesis=expected_genesis)
            audit = cls._validated_state_v3_audit(s["audit"])
            w.events = copy.deepcopy(s["events"])
            w.audit = audit
        else:
            w.events = copy.deepcopy(s["events"])
        for name in ("event_count", "history_hash", "first_assembly", "assemblies", "conversions",
                     "conversions_without_living_agent", "integrated_motif_time"):
            setattr(w, name, copy.deepcopy(s[name]))
        if schema != cls.plugin_schema:
            w.audit = copy.deepcopy(s["audit"])
        # Rebuild derived quantities using restored terrain, not generator assumptions.
        w._fertile_flat = [int(x) for x in np.flatnonzero(w.fertile)]
        w._channels[0] = ("source", w.config.source_rate*len(w._fertile_flat))
        w._bounds = np.cumsum([x[1] for x in w._channels]).tolist()
        w._total_rate = float(sum(x[1] for x in w._channels))
        w._update_field()
        if schema == cls.plugin_schema:
            if w._serialized_channels() != s["family"].get("channels"):
                raise ValueError("Plugin snapshot channel specification drift")
            derived = w._serialized_derived(w._derived_family_state)
            if derived != s["family"].get("derived"):
                raise ValueError("Plugin snapshot derived state drift")
            error = w.accounting_error()
            if abs(error["energy"]) > 1e-7 or error["material"] != 0:
                raise ValueError("Plugin snapshot conservation ledger is invalid")
        return w

    def resource_energy(self) -> float:
        return float(np.count_nonzero(self.resources == RAW)*self.config.raw_energy
                     + np.count_nonzero(self.resources == RICH)*self.config.rich_energy)

    def material_count(self) -> int:
        held = int(self.agent is not None and self.agent.inventory is not None) if hasattr(self, "agent") else 0
        return int(np.count_nonzero(self.resources)) + sum(p is not None for p in self.modules) + held

    def accounting_error(self) -> dict[str,float]:
        current = self.resource_energy() + (self.agent.energy if self.agent else 0.0)
        return dict(energy=self.audit["initial_energy"]+self.audit["external_energy"]-self.audit["dissipated_energy"]-current,
                    material=self.audit["initial_material"]+self.audit["incoming_material"]-self.audit["outgoing_material"]-self.material_count())

    def structural_match(self) -> bool:
        self._sync_legacy_family()
        self._derived_family_state = self._derive_family()
        self._field = set(self._derived_family_state.affected_locations)
        return self._derived_family_state.state.get("structural") is True

    def functional_motif(self) -> bool:
        self._sync_legacy_family()
        self._derived_family_state = self._derive_family()
        self._field = set(self._derived_family_state.affected_locations)
        return self._derived_family_state.functional

    def _update_field(self) -> None:
        self._sync_legacy_family()
        self._derived_family_state = self._derive_family()
        self._field = set(self._derived_family_state.affected_locations)

    def spawn(self, generation: int, *, energy: float | None = None,
              position: tuple[int,int] | None = None) -> None:
        if self.agent is not None:
            if self.agent.alive:
                raise ValueError("Cannot replace a living agent")
            self.retire()
        energy = self.config.initial_energy if energy is None else energy
        require_finite("birth energy", energy, positive=True)
        p = self.home if position is None else tuple(position)
        if not (0 <= p[0] < self.config.height and 0 <= p[1] < self.config.width):
            raise ValueError("Birth position out of bounds")
        self.agent = Agent(p, energy, self.time, generation=generation)
        self.audit["external_energy"] += energy
        self._log("birth", generation=generation, position=list(p), energy=energy)

    def _die(self, reason: str) -> None:
        if self.agent is None or not self.agent.alive:
            return
        a = self.agent
        a.alive = False; a.termination = reason
        # Private acquired notes do not survive; evaluator event history does,
        # but observe() does not expose it to any successor.
        a.memory = ""; a.last_result = {}
        if a.inventory is not None:
            taken = set(p for p in self.modules if p is not None)
            candidates = [a.position] + self.neighbors(a.position)
            drop = next((p for p in candidates if p not in taken), None)
            if drop is None:
                raise RuntimeError("No local site for carried component on death")
            self.modules[a.inventory] = drop
            self._log("death_drop", component=a.inventory, position=list(drop))
            a.inventory = None
            self._update_field()
        self._log("death", generation=a.generation, reason=reason, age=a.age, energy=a.energy)

    def retire(self) -> None:
        if self.agent is None:
            return
        if self.agent.alive:
            raise ValueError("Cannot retire a living individual")
        self.audit["dissipated_energy"] += self.agent.energy
        self._log("retired", generation=self.agent.generation, residual_energy=self.agent.energy)
        self.agent = None

    def _schedule(self) -> None:
        if self._pending is not None or self._total_rate == 0:
            return
        delay = float(self.rng.exponential(1.0/self._total_rate))
        choice = float(self.rng.random())*self._total_rate
        index = min(bisect.bisect_right(self._bounds, choice), len(self._channels)-1)
        name = self._channels[index][0]
        if name in self._family_channel_ids:
            spec = next(channel for channel in self._family_channels if channel.channel_id == name)
            target = 0
            accept = 0.0
            if DrawRequirement.TARGET_INDEX in spec.draw_requirements:
                u = float(self.rng.random())
                n = 3 if spec.target_domain is TargetDomain.MODULE else self.config.width*self.config.height
                target = min(int(u*n), n-1)
            if DrawRequirement.ACCEPTANCE_UNIFORM in spec.draw_requirements:
                accept = float(self.rng.random())
        else:
            u, accept = float(self.rng.random()), float(self.rng.random())
            n = len(self._fertile_flat) if name == "source" else (3 if name == "module_decay" else self.config.width*self.config.height)
            target = min(int(u*n), n-1)
            if name == "source":
                target = self._fertile_flat[target]
        self._pending = (self.time+delay, name, target, accept)

    def _transition_failure(self, message: str) -> None:
        exc = FamilyTransitionError(message)
        self._family_error("transition", exc)
        raise exc

    def _apply_family_transition(
        self, transition: LawTransition | InterventionTransition
    ) -> None:
        capabilities = transition.declared_capabilities
        if not capabilities.issubset(self._family.descriptor.capabilities):
            self._transition_failure("transition uses a capability not declared by the family")
        resources = self.resources.reshape(-1).astype(np.int8, copy=True)
        modules = list(self.modules)
        module_states = copy.deepcopy(self._module_states)
        evidence: list[EventEvidence] = []
        material_delta = 0
        energy_delta = 0.0
        cell_count = self.config.width * self.config.height

        try:
            for operation in transition.operations:
                required = getattr(operation, "required_capability", None)
                if required not in capabilities:
                    raise ValueError("transition operation capability is undeclared")
                if isinstance(operation, ResourceReplacement):
                    if operation.cell_index >= cell_count:
                        raise IndexError("resource target is out of bounds")
                    current = int(resources[operation.cell_index])
                    if current != operation.expected_value:
                        raise ValueError("resource target does not match expected current value")
                    if operation.replacement_value not in {EMPTY, RAW, RICH}:
                        raise ValueError("resource replacement value is outside the fixed vocabulary")
                    old_material = int(current != EMPTY)
                    new_material = int(operation.replacement_value != EMPTY)
                    energies = {
                        EMPTY: 0.0,
                        RAW: self.config.raw_energy,
                        RICH: self.config.rich_energy,
                    }
                    material_delta += new_material - old_material
                    energy_delta += energies[operation.replacement_value] - energies[current]
                    resources[operation.cell_index] = operation.replacement_value
                elif isinstance(operation, ResourcePreservation):
                    if operation.cell_index >= cell_count:
                        raise IndexError("resource preservation target is out of bounds")
                    if int(resources[operation.cell_index]) != operation.expected_value:
                        raise ValueError("resource preservation does not match expected current value")
                elif isinstance(operation, ModulePositionChange):
                    index = operation.module_index
                    if modules[index] != operation.expected_position:
                        raise ValueError("module position does not match expected current position")
                    replacement = operation.replacement_position
                    if replacement is not None:
                        if not (
                            0 <= replacement[0] < self.config.height
                            and 0 <= replacement[1] < self.config.width
                        ):
                            raise IndexError("module replacement position is out of bounds")
                        if any(
                            other_index != index and position == replacement
                            for other_index, position in enumerate(modules)
                        ):
                            raise ValueError("module replacement position is occupied")
                    material_delta += int(replacement is not None) - int(modules[index] is not None)
                    modules[index] = replacement
                elif isinstance(operation, ModuleStateChange):
                    index = operation.module_index
                    current = module_states[index].get(operation.state_key)
                    if freeze_json(current) != operation.expected_state:
                        raise ValueError("module state does not match expected current state")
                    module_states[index][operation.state_key] = thaw_json(operation.replacement_state)
                elif isinstance(operation, EventEvidence):
                    if operation.location_index is not None and operation.location_index >= cell_count:
                        raise IndexError("event evidence location is out of bounds")
                    evidence.append(operation)
                else:
                    raise TypeError("transition operation type is not in the closed vocabulary")
            if material_delta != transition.accounting.material_delta:
                raise ValueError("transition material accounting mismatch")
            if energy_delta != transition.accounting.energy_delta:
                raise ValueError("transition energy accounting mismatch")
        except (IndexError, TypeError, ValueError) as exc:
            self._transition_failure(str(exc))

        candidate_instance = (
            transition.resulting_instance
            if isinstance(transition, InterventionTransition)
            else self._family_instance
        )
        try:
            candidate_view = self._candidate_substrate_view(resources, modules, module_states)
            candidate_instance, private_record = self._synchronized_instance(
                candidate_view, candidate_instance
            )
            if (
                private_record is not None
                and len(self._private_transition_records)
                >= PLUGIN_PRIVATE_TRANSITION_RECORD_LIMIT
            ):
                raise ValueError("plugin private transition record limit exceeded")
            if private_record is not None and not self.record:
                raise ValueError(
                    "plugin private state transitions require record=True for auditability"
                )
            candidate_derived = self._family.derive(
                candidate_view,
                candidate_instance,
            )
            if not isinstance(candidate_derived, DerivedLawState):
                raise TypeError("LawFamily.derive() must return DerivedLawState")
        except Exception as exc:
            self._transition_failure(f"candidate derived state is invalid: {exc}")

        self.resources.reshape(-1)[:] = resources
        self.modules = modules
        self._module_states = module_states
        self._family_instance = candidate_instance
        self.mechanism_enabled = candidate_instance.enabled
        self._derived_family_state = candidate_derived
        self._field = set(candidate_derived.affected_locations)
        if private_record is not None:
            self._commit_private_transition_record(private_record)
        if energy_delta >= 0.0:
            self.audit["external_energy"] += energy_delta
        else:
            self.audit["dissipated_energy"] += -energy_delta
        if material_delta >= 0:
            self.audit["incoming_material"] += material_delta
        else:
            self.audit["outgoing_material"] += -material_delta
        for operation in evidence:
            self._log(
                "family_evidence",
                event=operation.event_type,
                target=operation.location_index,
                details=thaw_json(operation.details),
            )

    def _proposal(self, name: str, target: int, uniform: float) -> None:
        if (
            not self._legacy_mode
            and name in self._family_channel_ids
            and len(self._proposal_records) >= PLUGIN_PROPOSAL_RECORD_LIMIT
        ):
            # A saturated evaluator record stream is an administrative failure,
            # not a simulator transition. It must leave even error history exact.
            raise FamilyTransitionError("plugin proposal record limit exceeded")
        self.proposal_count += 1
        flat = self.resources.reshape(-1)
        c = self.config
        if name == "source":
            if flat[target] != EMPTY or (self.regime == 1 and uniform >= c.lean_source_multiplier):
                return
            flat[target] = RAW
            self.audit["external_energy"] += c.raw_energy
            self.audit["incoming_material"] += 1
        elif name == "raw_decay":
            if flat[target] != RAW:
                return
            atomic = {
                "audit": copy.deepcopy(self.audit),
                "derived": self._derived_family_state,
                "event_count": self.event_count,
                "event_length": len(self.events),
                "field": set(self._field),
                "history_hash": self.history_hash,
                "instance": self._family_instance,
                "mechanism_enabled": self.mechanism_enabled,
                "private_record_length": len(self._private_transition_records),
                "proposal_count": self.proposal_count - 1,
                "proposal_record_length": len(self._proposal_records),
                "resources": self.resources.copy(),
            }
            try:
                proposal = ProposalDraw(
                    "raw_decay", target, uniform, self.proposal_count, self.time,
                )
                derived = self._derive_family()
                rejection = self._call_family(
                    "filter_kernel_proposal", proposal, self._substrate_view(),
                    self._family_instance, derived,
                )
                if rejection is not None:
                    if len(self._proposal_records) >= PLUGIN_PROPOSAL_RECORD_LIMIT:
                        self._transition_failure("plugin proposal record limit exceeded")
                    if not isinstance(rejection, KernelProposalRejection) or rejection.proposal != proposal:
                        self._transition_failure("kernel proposal filter returned an invalid rejection")
                    if not rejection.declared_capabilities.issubset(self._family.descriptor.capabilities):
                        self._transition_failure("kernel proposal filter uses an undeclared capability")
                    evidence: list[EventEvidence] = []
                    for operation in rejection.operations:
                        if isinstance(operation, ResourcePreservation):
                            if operation.cell_index != target or operation.expected_value != RAW:
                                self._transition_failure("kernel proposal rejection preservation is invalid")
                        elif isinstance(operation, EventEvidence):
                            if (
                                operation.location_index is not None
                                and operation.location_index >= c.width * c.height
                            ):
                                self._transition_failure(
                                    "kernel proposal rejection evidence target is invalid"
                                )
                            evidence.append(operation)
                    # Encoding and capacity validation happen before evaluator history.
                    self._append_proposal_record(proposal, derived, rejection)
                    for operation in evidence:
                        self._log("family_evidence", event=operation.event_type,
                                  target=operation.location_index,
                                  details=thaw_json(operation.details))
                    return
                flat[target] = EMPTY
                self.audit["dissipated_energy"] += c.raw_energy
                self.audit["outgoing_material"] += 1
            except Exception:
                self.audit = atomic["audit"]
                self._derived_family_state = atomic["derived"]
                self.event_count = atomic["event_count"]
                del self.events[atomic["event_length"]:]
                self._field = atomic["field"]
                self.history_hash = atomic["history_hash"]
                self._family_instance = atomic["instance"]
                self.mechanism_enabled = atomic["mechanism_enabled"]
                del self._private_transition_records[atomic["private_record_length"]:]
                self.proposal_count = atomic["proposal_count"]
                del self._proposal_records[atomic["proposal_record_length"]:]
                self.resources = atomic["resources"]
                raise
        elif name == "rich_decay":
            if flat[target] != RICH:
                return
            flat[target] = RAW
            self.audit["dissipated_energy"] += c.rich_energy-c.raw_energy
        elif name in self._family_channel_ids:
            proposal = ProposalDraw(
                channel_id=name,
                target_index=target,
                acceptance_uniform=uniform,
                proposal_index=self.proposal_count,
                simulated_time=self.time,
            )
            derived = self._derive_family()
            transition = self._call_family(
                "apply_proposal",
                proposal,
                self._substrate_view(),
                self._family_instance,
                derived,
            )
            if transition is None:
                self._append_proposal_record(proposal, derived, None)
                return
            if not isinstance(transition, LawTransition):
                self._transition_failure("apply_proposal must return LawTransition or None")
            converted = sum(
                isinstance(operation, ResourceReplacement)
                and operation.expected_value == RAW
                and operation.replacement_value == RICH
                for operation in transition.operations
            )
            proposal_record = self._proposal_record(proposal, derived, transition)
            self._apply_family_transition(transition)
            if proposal_record is not None:
                self._proposal_records.append(proposal_record)
            self.conversions += converted
            if self.agent is None or not self.agent.alive:
                self.conversions_without_living_agent += converted
        elif name == "module_decay":
            held = self.agent is not None and self.agent.inventory == target
            if self.modules[target] is None and not held:
                return
            self.modules[target] = None
            if held:
                self.agent.inventory = None
            self.audit["outgoing_material"] += 1
            self._update_field()
        elif name == "regime":
            self.regime = 1-self.regime
        else:
            raise RuntimeError(f"Unknown proposal: {name}")
        self._log("physics", event=name, target=target)

    def advance(self, duration: float, *, stop_at_death: bool = False) -> None:
        require_finite("duration", duration)
        target = self.time+duration
        while self.time < target:
            self._schedule()
            death_at = math.inf
            reason = "lifespan"
            if self.agent and self.agent.alive:
                age_left = max(0.0, self.config.lifespan-self.agent.age)
                energy_left = self.agent.energy/self.config.metabolism if self.config.metabolism else math.inf
                death_at = self.time+min(age_left, energy_left)
                reason = "energy" if energy_left <= age_left else "lifespan"
            next_at = self._pending[0] if self._pending else math.inf
            derived = self._derive_family()
            deadline = self._call_family(
                "internal_deadline",
                self._substrate_view(),
                self._family_instance,
                derived,
            )
            if deadline is None:
                internal_at = math.inf
            elif (
                type(deadline) is not float
                or not math.isfinite(deadline)
                or deadline <= self.time
            ):
                self._transition_failure("family internal deadline is invalid")
            else:
                internal_at = deadline
            end = min(target, next_at, death_at, internal_at)
            dt = max(0.0, end-self.time)
            if derived.functional:
                self.integrated_motif_time += dt
            if self.agent and self.agent.alive:
                cost = min(self.agent.energy, dt*self.config.metabolism)
                self.agent.energy -= cost; self.agent.age += dt
                self.audit["dissipated_energy"] += cost
            self.time = end
            if death_at <= end:
                self._die(reason)
                if stop_at_death:
                    return
            if next_at <= end and self._pending is not None:
                _, name, index, uniform = self._pending
                self._proposal(name, index, uniform)
                self._pending = None
            if end >= target:
                return

    @staticmethod
    def normalize_action(action: Any, config: Config) -> tuple[dict[str,Any], bool]:
        if not isinstance(action, dict) or type(action.get("type")) is not str:
            return {"type":"WAIT", "duration":config.default_wait}, False
        kind = action["type"]
        if kind not in {"MOVE","PICK","DROP","CONSUME","WAIT"}:
            return {"type":"WAIT", "duration":config.default_wait}, False
        if kind == "MOVE":
            if action.get("direction") not in DIRECTIONS or set(action)-{"type","direction"}:
                return {"type":"WAIT", "duration":config.default_wait}, False
            return {"type":kind, "direction":action["direction"]}, True
        if kind == "WAIT":
            duration = action.get("duration", config.default_wait)
            try:
                require_finite("wait", duration, positive=True)
            except ValueError:
                return {"type":"WAIT", "duration":config.default_wait}, False
            if duration < 0.1 or duration > config.max_wait or set(action)-{"type","duration"}:
                return {"type":"WAIT", "duration":config.default_wait}, False
            return {"type":kind, "duration":float(duration)}, True
        if set(action) != {"type"}:
            return {"type":"WAIT", "duration":config.default_wait}, False
        return {"type":kind}, True

    def observe(self) -> dict[str,Any]:
        if self.agent is None or not self.agent.alive:
            raise RuntimeError("No living observer")
        a,c = self.agent,self.config
        cells = []
        for y in range(max(0,a.position[0]-c.radius), min(c.height,a.position[0]+c.radius+1)):
            for x in range(max(0,a.position[1]-c.radius), min(c.width,a.position[1]+c.radius+1)):
                items = []
                resource = int(self.resources[y,x])
                if resource:
                    items.append({"id":self.symbols[resource-1], "consume":True, "pick":False})
                for i,p in enumerate(self.modules):
                    if p == (y,x):
                        items.append({"id":self.symbols[i+2], "consume":False, "pick":True})
                cells.append({"position":[y,x], "surface":int(self.fertile[y,x]), "objects":items})
        current_cell=copy.deepcopy(next(cell for cell in cells if cell["position"]==list(a.position)))
        legal_directions=[]
        for direction,(dy,dx) in DIRECTIONS.items():
            y,x=a.position[0]+dy,a.position[1]+dx
            if 0<=y<c.height and 0<=x<c.width:
                legal_directions.append(direction)
        legal_actions={
            "MOVE":{"directions":legal_directions},
            "PICK":{"available":a.inventory is None and any(p==a.position for p in self.modules)},
            "DROP":{"available":a.inventory is not None and a.position not in self.modules},
            "CONSUME":{"available":bool(self.resources[a.position])},
            "WAIT":{"duration_min":0.1,"duration_max":c.max_wait}}
        # No seed, active pair, latent material IDs, global events, mechanism flag,
        # fertile rates, or evaluator measurements cross this boundary.
        observation = dict(time=round(self.time,8), age=round(a.age,8), remaining=round(c.lifespan-a.age,8),
                    energy=round(a.energy,8), position=list(a.position), bounds=[c.height,c.width],
                    inventory=self.symbols[a.inventory+2] if a.inventory is not None else None,
                    inventory_state={"occupied":a.inventory is not None,
                                     "object_id":self.symbols[a.inventory+2] if a.inventory is not None else None},
                    coordinate_system={"position":"[row,column]","N":"row-1","E":"column+1",
                                       "S":"row+1","W":"column-1"},
                    current_cell=current_cell,legal_actions=legal_actions,
                    local=cells, last_result=copy.deepcopy(a.last_result), memory=a.memory,
                    actions={"MOVE":{"direction":list(DIRECTIONS)}, "PICK":{}, "DROP":{}, "CONSUME":{},
                             "WAIT":{"duration_min":0.1,"duration_max":c.max_wait}},
                    costs={"cognition_energy":c.cognition_energy,"cognition_time":c.cognition_time,
                           "metabolism":c.metabolism,"move_time":c.move_time,"manipulate_time":c.manipulate_time,
                           "consume_time":c.consume_time})
        derived = self._derive_family()
        projection = self._call_family(
            "project_public", self._public_view(), self._family_instance, derived
        )
        try:
            frozen = freeze_json(projection, path="law_observation")
            if not isinstance(frozen, Mapping):
                raise TypeError("project_public() must return a JSON object")
            schema = self._family.descriptor.observation_schema
            _validate_projection(frozen, schema)
        except (TypeError, ValueError) as exc:
            self._family_error("project_public", exc)
            raise FamilyCallbackError("Invalid family public projection") from exc
        if frozen:
            observation["law_observation"] = thaw_json(frozen)
        return observation

    def step(self, decision: Any) -> dict[str,Any]:
        if self.agent is None or not self.agent.alive:
            raise RuntimeError("Cannot act after termination")
        a,c = self.agent,self.config
        valid_envelope = isinstance(decision,dict) and isinstance(decision.get("memory",""),str)
        action = decision.get("action") if isinstance(decision,dict) else None
        action, valid = self.normalize_action(action,c)
        valid = valid and valid_envelope and not bool(decision.get("invalid",False) if isinstance(decision,dict) else True)
        a.decisions += 1; a.invalid_actions += int(not valid)
        if isinstance(decision,dict):
            a.memory = str(decision.get("memory", ""))[:c.private_memory_chars]
        start_energy,start_time = a.energy,self.time
        self._log("decision", action=action, valid=valid, generation=a.generation)
        charge = min(a.energy,c.cognition_energy)
        a.energy -= charge; self.audit["dissipated_energy"] += charge
        if a.energy <= 1e-12:
            self._die("cognition")
            return {"status":"terminated", "action":action}
        self.advance(c.cognition_time,stop_at_death=True)
        if not a.alive:
            return {"status":"terminated", "action":action}
        kind = action["type"]
        duration = {"MOVE":c.move_time,"PICK":c.manipulate_time,"DROP":c.manipulate_time,
                    "CONSUME":c.consume_time,"WAIT":action.get("duration",c.default_wait)}[kind]
        self.advance(duration,stop_at_death=True)
        if not a.alive:
            return {"status":"terminated", "action":action}
        structural_before = (
            self.functional_motif() if self._legacy_mode else self.structural_match()
        )
        status = "no_effect"; gross = 0.0; object_id = None
        if kind == "MOVE":
            dy,dx = DIRECTIONS[action["direction"]]
            y,x = a.position[0]+dy,a.position[1]+dx
            if 0 <= y < c.height and 0 <= x < c.width:
                a.position = (y,x); status = "moved"
            else:
                status = "blocked"
        elif kind == "PICK":
            if a.inventory is None:
                match = next((i for i,p in enumerate(self.modules) if p == a.position),None)
                if match is not None:
                    a.inventory = match; self.modules[match] = None; status = "picked"
                    object_id = self.symbols[match+2]
        elif kind == "DROP":
            if a.inventory is not None and a.position not in self.modules:
                object_id = self.symbols[a.inventory+2]
                self.modules[a.inventory] = a.position; a.inventory = None; status = "dropped"
        elif kind == "CONSUME":
            value = int(self.resources[a.position])
            if value:
                gross = c.raw_energy if value == RAW else c.rich_energy
                object_id = self.symbols[value-1]
                self.resources[a.position] = EMPTY
                a.energy += gross
                a.raw_consumed += int(value==RAW); a.rich_consumed += int(value==RICH)
                self.audit["outgoing_material"] += 1
                status = "consumed"
        else:
            status = "waited"
        self._update_field()
        structural_after = (
            self.functional_motif()
            if self._legacy_mode
            else self._derived_family_state.state.get("structural") is True
        )
        if structural_after and not structural_before and kind == "DROP" and status == "dropped":
            self.assemblies += 1
            if self.first_assembly is None:
                self.first_assembly = self.time
            self._log("assembly", generation=a.generation)
        result = dict(action=action,status=status,object_id=object_id,gross_energy=gross,
                      energy_change=a.energy-start_energy,elapsed=self.time-start_time,valid=valid)
        a.last_result = result
        self._log("action", generation=a.generation, **result, position=list(a.position), energy=a.energy)
        return copy.deepcopy(result)

    def apply_control(self, control: ControlKind) -> None:
        control = ControlKind(control)
        if control not in {ControlKind.NULL, ControlKind.KNOCKOUT}:
            raise ValueError("apply_control supports only mechanism-disabled controls")
        self._validated_controls()
        transition = self._call_family(
            "intervene",
            control,
            self._substrate_view(),
            self._family_instance,
        )
        if not isinstance(transition, InterventionTransition) or transition.control is not control:
            self._transition_failure(f"{control.value} intervention returned an invalid transition")
        result = transition.resulting_instance
        self._validate_resulting_instance(result)
        if not self._family_instance.enabled or result.enabled:
            self._transition_failure(
                f"{control.value} must change an enabled instance to disabled"
            )
        if (result.hidden_parameters != self._family_instance.hidden_parameters
                or result.private_state != self._family_instance.private_state):
            self._transition_failure(
                f"{control.value} must preserve hidden and private family identity"
            )
        if (transition.operations or transition.declared_capabilities
                or transition.accounting != AccountingDelta()):
            self._transition_failure(
                f"{control.value} must have zero operations and accounting"
            )
        try:
            result_channels = validate_channel_specs(self._family.channels(result, self.config))
        except Exception as exc:
            self._transition_failure(f"{control.value} channels are invalid: {exc}")
        if result_channels != self._family_channels:
            self._transition_failure(
                f"{control.value} must preserve channel specifications"
            )
        self._apply_family_transition(transition)
        self._family_instance = result
        self.mechanism_enabled = self._family_instance.enabled
        self._log(
            "intervention",
            intervention=(
                "mechanism_knockout"
                if control is ControlKind.KNOCKOUT
                else "matched_null"
            ),
        )
        # Do not alter pending proposals, rates, or RNG state.

    def knockout(self) -> None:
        self.apply_control(ControlKind.KNOCKOUT)

    def break_geometry(self) -> bool:
        if not self.structural_match():
            return False
        self._validated_controls()
        transition = self._call_family(
            "intervene",
            ControlKind.BROKEN,
            self._substrate_view(),
            self._family_instance,
        )
        if not isinstance(transition, InterventionTransition) or transition.control is not ControlKind.BROKEN:
            self._transition_failure("broken intervention returned an invalid transition")
        self._validate_resulting_instance(transition.resulting_instance)
        if transition.resulting_instance != self._family_instance:
            self._transition_failure("broken control must preserve the family instance")
        if transition.accounting != AccountingDelta():
            self._transition_failure("broken control must preserve matter and energy")
        changes = [
            operation
            for operation in transition.operations
            if isinstance(operation, ModulePositionChange)
        ]
        if len(transition.operations) != 1 or len(changes) != 1:
            self._transition_failure("broken intervention must move exactly one module")
        change = changes[0]
        candidate_modules = list(self.modules)
        if candidate_modules[change.module_index] != change.expected_position:
            self._transition_failure("broken module position does not match current position")
        candidate_modules[change.module_index] = change.replacement_position
        try:
            candidate_derived = self._family.derive(
                self._candidate_substrate_view(
                    self.resources.reshape(-1).astype(np.int8, copy=True),
                    candidate_modules,
                    copy.deepcopy(self._module_states),
                ),
                self._family_instance,
            )
            if not isinstance(candidate_derived, DerivedLawState):
                raise TypeError("LawFamily.derive() must return DerivedLawState")
        except Exception as exc:
            self._transition_failure(f"broken candidate derived state is invalid: {exc}")
        if candidate_derived.state.get("structural") is True:
            self._transition_failure("broken intervention did not disrupt the declared structure")
        self._apply_family_transition(transition)
        self.mechanism_enabled = self._family_instance.enabled
        self._log(
            "intervention",
            intervention="geometry_broken",
            component=change.module_index,
            old=list(change.expected_position),
            new=list(change.replacement_position),
        )
        return True

    def normalize_resources(self, reference: np.ndarray) -> None:
        arr = np.asarray(reference, dtype=np.int8)
        if arr.shape != self.resources.shape or np.any((arr<0)|(arr>2)):
            raise ValueError("Invalid resource intervention")
        before_energy,before_material = self.resource_energy(),int(np.count_nonzero(self.resources))
        self.resources = arr.copy()
        delta = self.resource_energy()-before_energy
        self.audit["external_energy"] += max(0.0,delta)
        self.audit["dissipated_energy"] += max(0.0,-delta)
        material_delta = int(np.count_nonzero(arr))-before_material
        self.audit["incoming_material"] += max(0,material_delta)
        self.audit["outgoing_material"] += max(0,-material_delta)
        self._log("intervention", intervention="stock_normalization", energy_delta=delta, material_delta=material_delta)

    def evaluate_family_evidence(
        self, policy_records: tuple[Mapping[str, Any], ...] = (), *,
        evaluator_event_start: int = 0,
        evaluator_baseline: Mapping[str, Any] | None = None,
    ) -> FamilyEvidence:
        """Evaluate standardized evidence from a fixed detached evaluator trace."""

        if (
            type(evaluator_event_start) is not int
            or not 0 <= evaluator_event_start <= len(self.events)
        ):
            raise ValueError("Evaluator event start is invalid")
        baseline_fields = {
            "evaluator_event_start", "proposal_record_start", "initial_functional",
            "initial_assemblies", "initial_conversions", "initial_proposals",
            "initial_time",
        }
        windowed = evaluator_baseline is not None
        if evaluator_baseline is None:
            baseline = {
                "evaluator_event_start": evaluator_event_start,
                "proposal_record_start": 0,
                "initial_functional": False,
                "initial_assemblies": 0,
                "initial_conversions": 0,
                "initial_proposals": 0,
                "initial_time": 0.0,
            }
        else:
            if type(evaluator_baseline) is not dict or set(evaluator_baseline) != baseline_fields:
                raise ValueError("Evaluator baseline fields are invalid")
            self._validate_exact_json(evaluator_baseline, path="Evaluator baseline")
            baseline = copy.deepcopy(evaluator_baseline)
            if baseline["evaluator_event_start"] != evaluator_event_start:
                raise ValueError("Evaluator baseline event start is inconsistent")
        base_evaluator_events = [
            copy.deepcopy(event) for event in self.events[evaluator_event_start:]
        ]
        evaluator_events: list[dict[str, Any]] = []
        insertions: dict[int, list[dict[str, Any]]] = {}
        prior_observation_offset = 0
        prior_result_offset = 0
        for index, record in enumerate(policy_records):
            if not isinstance(record, Mapping):
                raise ValueError("Policy evidence records have invalid fields")
            if set(record) == {"observation", "result"}:
                observation_offset = len(base_evaluator_events)
                result_offset = len(base_evaluator_events)
            elif set(record) == {
                "observation", "observation_event_offset", "result",
                "result_event_offset",
            }:
                observation_offset = record["observation_event_offset"]
                result_offset = record["result_event_offset"]
                if (
                    type(observation_offset) is not int
                    or type(result_offset) is not int
                    or not 0 <= observation_offset <= result_offset <= len(base_evaluator_events)
                    or observation_offset < prior_observation_offset
                    or result_offset < prior_result_offset
                ):
                    raise ValueError("Policy evidence record ordering is invalid")
            else:
                raise ValueError("Policy evidence records have invalid fields")
            insertions.setdefault(observation_offset, []).append({
                "kind": "policy_observation",
                "decision_index": index,
                "observation": copy.deepcopy(record["observation"]),
            })
            insertions.setdefault(result_offset, []).append({
                "kind": "policy_result",
                "decision_index": index,
                "result": copy.deepcopy(record["result"]),
            })
            prior_observation_offset = observation_offset
            prior_result_offset = result_offset
        for offset in range(len(base_evaluator_events) + 1):
            evaluator_events.extend(insertions.get(offset, ()))
            if offset < len(base_evaluator_events):
                evaluator_events.append(base_evaluator_events[offset])
        terminal = {
            "agent": None if self.agent is None else {
                "alive": self.agent.alive,
                "generation": self.agent.generation,
                "raw_consumed": self.agent.raw_consumed,
                "rich_consumed": self.agent.rich_consumed,
                "termination": self.agent.termination,
            },
            "assemblies": (
                self.assemblies - baseline["initial_assemblies"]
                if windowed else self.assemblies
            ),
            "conversions": (
                self.conversions - baseline["initial_conversions"]
                if windowed else self.conversions
            ),
            "family_id": self._family.descriptor.family_id,
            "functional": self.functional_motif(),
            "module_positions": [list(position) if position is not None else None for position in self.modules],
            "proposal_count": (
                self.proposal_count - baseline["initial_proposals"]
                if windowed else self.proposal_count
            ),
            "raw_symbol": self.symbols[0],
            "rich_symbol": self.symbols[1],
            "simulated_time": self.time,
            "width": self.config.width,
            **baseline,
        }
        value = self._call_family(
            "evaluate", EvaluatorTrace(tuple(evaluator_events), terminal)
        )
        if not isinstance(value, FamilyEvidence):
            exc = TypeError("LawFamily.evaluate() must return FamilyEvidence")
            self._family_error("evaluate", exc)
            raise FamilyCallbackError(str(exc))
        if any(reference >= len(evaluator_events) for reference in value.event_references):
            exc = ValueError("FamilyEvidence event reference is out of bounds")
            self._family_error("evaluate", exc)
            raise FamilyCallbackError(str(exc))
        return value

    def display_state(self) -> dict[str,Any]:
        """Evaluator/UI only. Must never be passed to the policy."""
        return dict(time=self.time,resources=self.resources.tolist(),modules=[list(p) if p is not None else None for p in self.modules],
                    fertile=self.fertile.astype(int).tolist(),home=list(self.home),
                    agent=asdict(self.agent) if self.agent else None,motif=self.functional_motif(),
                    conversions=self.conversions,first_assembly=self.first_assembly,
                    active_pair=list(self.law.pair),symbols=self.symbols,regime=self.regime)
