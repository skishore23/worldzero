"""Deterministic validation kit for trusted in-process law-family plugins.

The kit is evaluator-side code.  It deliberately exercises a selected plugin
through the real fixed kernel, snapshots, controls, and trace-v4 replay while
turning plugin contract failures into finite machine-readable JSON.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from itertools import permutations
import random
import re
import sys
from collections.abc import Iterable, Mapping, Sequence, Set
from types import (
    BuiltinFunctionType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MemberDescriptorType,
    MethodType,
    ModuleType,
    SimpleNamespace,
)
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from worldzero.util import canonical

if TYPE_CHECKING:
    from worldzero.kernel import World

from .registry import (
    LawRegistry,
    RegisteredFamily,
    calibration_suite_fingerprint,
)
from .base import LawFamily
from .types import (
    AccountingDelta,
    CalibrationCase,
    ChannelSpec,
    ControlKind,
    ControlSuite,
    DerivedLawState,
    DrawRequirement,
    EvaluatorTrace,
    FamilyDescriptor,
    FamilyEvidence,
    FamilyInstance,
    InterventionTransition,
    KernelProposalRejection,
    LawTransition,
    ModulePositionChange,
    ProposalDraw,
    PrivateStateTransition,
    PublicSubstrateView,
    SampleContext,
    SubstrateView,
    TargetDomain,
    thaw_json,
    validate_channel_specs,
)


REPORT_SCHEMA = "worldzero-family-validation-v1"
_CALLBACK_RECORDS = (
    DerivedLawState,
    EvaluatorTrace,
    FamilyInstance,
    ProposalDraw,
    PublicSubstrateView,
    SampleContext,
    SubstrateView,
)
_REQUIRED_MATCHING_CONSTRAINTS = frozenset({
    "material_stock", "proposal_stream", "public_substrate",
})
_COMMUNITY_CALIBRATION_CONTRACTS = {
    "deterministic_callbacks": (
        "ambient_rng", "callback_isolation", "determinism",
        "state_independent_channels",
    ),
    "lifecycle": ("lifecycle",),
    "matched_controls": ("controls",),
    "observation_boundary": ("observation_boundary",),
    "snapshot_replay": ("snapshot_replay",),
    "transition_accounting": ("accounting",),
}
_MAX_COMMUNITY_CALIBRATION_SAMPLES = 4096
_OBJECT_GRAPH_MAX_NODES = 4096
_OBJECT_GRAPH_MAX_DEPTH = 32
_OBJECT_GRAPH_MAX_BYTES = 1024 * 1024
_STORAGE_EXCLUDED_TYPES = (
    ModuleType,
    type,
    FunctionType,
    BuiltinFunctionType,
    MethodType,
    property,
    staticmethod,
    classmethod,
    MemberDescriptorType,
    GetSetDescriptorType,
)
_PRIVATE_RNG_TYPES = (random.Random, np.random.Generator, np.random.RandomState)
_FORBIDDEN_OBSERVATION_TERMS = frozenset({
    "active_pair",
    "calibration",
    "control_arm",
    "enabled",
    "evaluator",
    "family_id",
    "fingerprint",
    "future",
    "hidden",
    "mechanism_enabled",
    "pair_id",
    "proposal_index",
})


def _numpy_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _bounded_message(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    text = re.sub(r"0x[0-9a-fA-F]+", "0x<address>", text)
    return text[:300]


class ObjectGraphLimitError(RuntimeError):
    """Plugin-owned state exceeded the validator's finite inspection budget."""


class ObjectGraphUnsupportedError(RuntimeError):
    """Plugin-owned mutable state cannot be inspected without arbitrary code."""


@dataclass(frozen=True)
class _StorageRootFingerprint:
    identity: int
    content_sha256: str | None
    private_rng: bool
    failure: BaseException | None = None


def _graph_node_bytes(value: object) -> int:
    size = 64
    if isinstance(value, str):
        size += len(value.encode("utf-8", errors="replace"))
    elif isinstance(value, (bytes, bytearray, memoryview)):
        size += len(value)
    elif isinstance(value, np.ndarray):
        size += int(value.nbytes)
    return size


def _raw_instance_dict(value: object) -> Mapping[str, object]:
    """Read only the concrete built-in instance-dictionary descriptor."""

    for value_type in type(value).__mro__:
        descriptor = vars(value_type).get("__dict__")
        if isinstance(descriptor, (GetSetDescriptorType, MemberDescriptorType)):
            try:
                storage = descriptor.__get__(value, type(value))
            except (AttributeError, TypeError):
                return {}
            return storage if isinstance(storage, Mapping) else {}
    return {}


def _raw_slot_items(value: object) -> tuple[tuple[str, object], ...]:
    """Read only statically declared CPython slot member descriptors."""

    items: list[tuple[str, object]] = []
    for value_type in type(value).__mro__:
        declared = vars(value_type).get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        if not isinstance(declared, (tuple, list)):
            continue
        for name in declared:
            if not isinstance(name, str) or name in {"__dict__", "__weakref__"}:
                continue
            descriptor = vars(value_type).get(name)
            if not isinstance(descriptor, MemberDescriptorType):
                continue
            try:
                item = descriptor.__get__(value, type(value))
            except AttributeError:
                continue
            items.append((name, item))
    return tuple(items)


def _safe_custom_storage(value: object, plugin_module: str) -> bool:
    value_module = type(value).__module__
    plugin_package = plugin_module.partition(".")[0]
    return (
        isinstance(value, SimpleNamespace)
        or value_module == plugin_module
        or (
            bool(plugin_package)
            and value_module.startswith(f"{plugin_package}.")
        )
    )


def _object_graph_children(value: object, plugin_module: str) -> tuple[object, ...]:
    if isinstance(value, dict):
        return tuple(item for pair in dict.items(value) for item in pair)
    if isinstance(value, MappingProxyType):
        return tuple(
            item for pair in MappingProxyType.items(value) for item in pair
        )
    if isinstance(value, list):
        return tuple(list.__iter__(value))
    if isinstance(value, tuple):
        return tuple(tuple.__iter__(value))
    if isinstance(value, set):
        return tuple(set.__iter__(value))
    if isinstance(value, frozenset):
        return tuple(frozenset.__iter__(value))
    if not (
        is_dataclass(value) and not isinstance(value, type)
    ) and not _safe_custom_storage(value, plugin_module):
        return ()
    children: list[object] = []
    children.extend(_raw_instance_dict(value).values())
    children.extend(item for _, item in _raw_slot_items(value))
    return tuple(children)


def _typed_semantic_key(
    value: object,
    plugin_module: str,
    *,
    seen: set[int],
    depth: int,
) -> object:
    """Return an injective in-process key token without string coercion."""

    if depth > _OBJECT_GRAPH_MAX_DEPTH:
        raise ObjectGraphLimitError(
            f"plugin-owned object graph exceeded depth limit {_OBJECT_GRAPH_MAX_DEPTH}"
        )
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        token: list[object] = ["float", float.hex(value)]
        if value != value:
            token.append(id(value))
        return token
    if type(value) is str:
        return ["str", value]
    if type(value) is bytes:
        return ["bytes", bytes.hex(value)]
    if isinstance(value, Enum):
        return [
            "enum",
            f"{type(value).__module__}.{type(value).__qualname__}",
            _typed_semantic_key(
                value.value, plugin_module, seen=seen, depth=depth + 1,
            ),
        ]
    identity = id(value)
    if identity in seen:
        return [
            "cycle-object",
            f"{type(value).__module__}.{type(value).__qualname__}",
            identity,
        ]
    if isinstance(value, tuple):
        seen.add(identity)
        try:
            return [
                "tuple",
                [
                    _typed_semantic_key(
                        item, plugin_module, seen=seen, depth=depth + 1,
                    )
                    for item in tuple.__iter__(value)
                ],
            ]
        finally:
            seen.remove(identity)
    if isinstance(value, frozenset):
        seen.add(identity)
        try:
            items = [
                _typed_semantic_key(
                    item, plugin_module, seen=seen, depth=depth + 1,
                )
                for item in frozenset.__iter__(value)
            ]
        finally:
            seen.remove(identity)
        items.sort(
            key=lambda item: json.dumps(
                item, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
            )
        )
        return ["frozenset", items]
    if _safe_custom_storage(value, plugin_module):
        return [
            "object",
            f"{type(value).__module__}.{type(value).__qualname__}",
            identity,
            _semantic_payload(
                value, plugin_module, seen=seen, depth=depth + 1,
            ),
        ]
    raise TypeError(
        f"mapping key type {type(value).__module__}.{type(value).__qualname__} "
        "has no safe typed representation"
    )


def _semantic_mapping_payload(
    pairs: Iterable[tuple[object, object]],
    plugin_module: str,
    *,
    seen: set[int],
    depth: int,
) -> dict[str, object]:
    rows = [
        {
            "key": _typed_semantic_key(
                key, plugin_module, seen=seen, depth=depth + 1,
            ),
            "value": _semantic_payload(
                item, plugin_module, seen=seen, depth=depth + 1,
            ),
        }
        for key, item in pairs
    ]
    rows.sort(
        key=lambda row: json.dumps(
            row["key"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return {"$mapping": rows}


def _semantic_payload(
    value: object, plugin_module: str, *, seen: set[int] | None = None,
    depth: int = 0,
) -> object:
    if depth > _OBJECT_GRAPH_MAX_DEPTH:
        raise ObjectGraphLimitError(
            f"plugin-owned object graph exceeded depth limit {_OBJECT_GRAPH_MAX_DEPTH}"
        )
    if value is None or isinstance(value, (str, bool)) or type(value) in {int, float}:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {"$bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (bytearray, memoryview)):
        return {"$bytes_sha256": hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value, _PRIVATE_RNG_TYPES):
        return {"$private_rng": f"{type(value).__module__}.{type(value).__qualname__}"}
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return {"$cycle": f"{type(value).__module__}.{type(value).__qualname__}"}
    seen.add(identity)
    try:
        if isinstance(value, dict):
            return _semantic_mapping_payload(
                dict.items(value), plugin_module, seen=seen, depth=depth,
            )
        if isinstance(value, MappingProxyType):
            return _semantic_mapping_payload(
                MappingProxyType.items(value),
                plugin_module,
                seen=seen,
                depth=depth,
            )
        named_fields = vars(type(value)).get("_fields")
        if isinstance(value, tuple) and isinstance(named_fields, tuple) and all(
            isinstance(name, str) for name in named_fields
        ):
            return _semantic_mapping_payload(
                zip(named_fields, tuple.__iter__(value), strict=True),
                plugin_module,
                seen=seen,
                depth=depth,
            )
        if isinstance(value, list):
            return [
                _semantic_payload(
                    item, plugin_module, seen=seen, depth=depth + 1,
                )
                for item in list.__iter__(value)
            ]
        if isinstance(value, tuple):
            return [
                _semantic_payload(
                    item, plugin_module, seen=seen, depth=depth + 1,
                )
                for item in tuple.__iter__(value)
            ]
        if isinstance(value, set):
            return sorted(
                (
                    _semantic_payload(
                        item, plugin_module, seen=seen, depth=depth + 1,
                    )
                    for item in set.__iter__(value)
                ),
                key=repr,
            )
        if isinstance(value, frozenset):
            return sorted(
                (
                    _semantic_payload(
                        item, plugin_module, seen=seen, depth=depth + 1,
                    )
                    for item in frozenset.__iter__(value)
                ),
                key=repr,
            )
        if (
            is_dataclass(value) and not isinstance(value, type)
        ) or _safe_custom_storage(value, plugin_module):
            storage = dict(_raw_instance_dict(value))
            for name, item in _raw_slot_items(value):
                storage.setdefault(name, item)
            return _semantic_mapping_payload(
                dict.items(storage), plugin_module, seen=seen, depth=depth,
            )
        raise TypeError("value has no safe semantic callback representation")
    finally:
        seen.remove(identity)

def _semantic_mapping_rows(value: object) -> list[object] | None:
    if not isinstance(value, dict) or set(value) != {"$mapping"}:
        return None
    rows = value["$mapping"]
    return rows if isinstance(rows, list) else None


def _semantic_contains(candidate: object, sensitive: object) -> bool:
    if candidate == sensitive:
        return True
    candidate_rows = _semantic_mapping_rows(candidate)
    sensitive_rows = _semantic_mapping_rows(sensitive)
    if candidate_rows is not None and sensitive_rows is not None:
        if all(
            isinstance(sensitive_row, dict)
            and any(
                isinstance(candidate_row, dict)
                and candidate_row.get("key") == sensitive_row.get("key")
                and candidate_row.get("value") == sensitive_row.get("value")
                for candidate_row in candidate_rows
            )
            for sensitive_row in sensitive_rows
        ):
            return True
    if isinstance(candidate, Mapping):
        return any(_semantic_contains(item, sensitive) for item in candidate.values())
    if isinstance(candidate, list):
        return any(_semantic_contains(item, sensitive) for item in candidate)
    return False


def _module_root_is_traversable(value: object, plugin_module: str) -> bool:
    if isinstance(value, _STORAGE_EXCLUDED_TYPES):
        return False
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex, Enum)):
        return False
    return (
        isinstance(value, (Mapping, Sequence, Set, bytearray, memoryview, np.ndarray))
        or is_dataclass(value) and not isinstance(value, type)
        or _safe_custom_storage(value, plugin_module)
        or isinstance(value, _PRIVATE_RNG_TYPES)
    )


def _storage_content_fingerprint(
    value: object, plugin_module: str,
) -> tuple[str, bool]:
    nodes, _ = _walk_object_graph((value,), plugin_module)
    private_rng = any(isinstance(node, _PRIVATE_RNG_TYPES) for node in nodes)
    try:
        payload = _semantic_payload(value, plugin_module)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except ObjectGraphLimitError:
        raise
    except Exception as exc:
        raise ObjectGraphUnsupportedError(
            "mutable root has no safe static semantic representation"
        ) from exc
    if len(encoded) > _OBJECT_GRAPH_MAX_BYTES:
        raise ObjectGraphLimitError(
            f"plugin-owned object graph exceeded byte limit {_OBJECT_GRAPH_MAX_BYTES}"
        )
    return hashlib.sha256(encoded).hexdigest(), private_rng


def _module_namespace_snapshot(
    family: object,
) -> dict[str, _StorageRootFingerprint]:
    plugin_module = type(family).__module__
    module = sys.modules.get(plugin_module)
    if module is None:
        return {}
    snapshot: dict[str, _StorageRootFingerprint] = {}
    for name, value in vars(module).items():
        if name.startswith("__") or not _module_root_is_traversable(
            value, plugin_module,
        ):
            continue
        try:
            content_sha256, private_rng = _storage_content_fingerprint(
                value, plugin_module,
            )
        except (ObjectGraphLimitError, ObjectGraphUnsupportedError) as exc:
            failure = type(exc)(
                f"module global {name!r} cannot be fingerprinted: {exc}"
            )
            snapshot[name] = _StorageRootFingerprint(
                id(value), None, False, failure,
            )
        except Exception as exc:
            failure = ObjectGraphUnsupportedError(
                f"module global {name!r} cannot be fingerprinted safely: "
                f"{type(exc).__name__}: {_bounded_message(exc)}"
            )
            snapshot[name] = _StorageRootFingerprint(
                id(value), None, False, failure,
            )
        else:
            snapshot[name] = _StorageRootFingerprint(
                id(value), content_sha256, private_rng,
            )
    return snapshot


def _root_fingerprint_changed(
    previous: _StorageRootFingerprint | None,
    current: _StorageRootFingerprint,
) -> bool:
    if previous is None:
        return True
    previous_overflow = isinstance(previous.failure, ObjectGraphLimitError)
    current_overflow = isinstance(current.failure, ObjectGraphLimitError)
    previous_error = previous.failure is not None
    current_error = current.failure is not None
    return (
        previous.identity != current.identity
        or previous.content_sha256 != current.content_sha256
        or previous.private_rng != current.private_rng
        or previous_overflow != current_overflow
        or previous_error != current_error
        or type(previous.failure) is not type(current.failure)
        or (
            previous.failure is not None
            and current.failure is not None
            and _bounded_message(previous.failure)
            != _bounded_message(current.failure)
        )
    )


def _family_storage_roots(
    family: object, module_before: Mapping[str, _StorageRootFingerprint],
) -> tuple[tuple[object, ...], tuple[BaseException, ...]]:
    """Read direct plugin storage without evaluating arbitrary descriptors."""

    roots: list[object] = []
    roots.extend(_raw_instance_dict(family).values())
    roots.extend(
        item for name, item in _raw_slot_items(family)
        if name not in {"_family", "_fail", "_seed"}
    )
    for family_type in type(family).__mro__:
        if family_type is LawFamily:
            break
        for name, value in vars(family_type).items():
            if name.startswith("__") or isinstance(value, _STORAGE_EXCLUDED_TYPES):
                continue
            roots.append(value)
    failures: list[BaseException] = []
    module = sys.modules.get(type(family).__module__)
    if module is not None:
        module_after = _module_namespace_snapshot(family)
        for name, value in vars(module).items():
            current = module_after.get(name)
            if current is None:
                continue
            if current.failure is not None:
                failures.append(current.failure)
            previous = module_before.get(name)
            if _root_fingerprint_changed(previous, current):
                roots.append(value)
    return tuple(roots), tuple(failures)


def _walk_object_graph(
    roots: Iterable[object], plugin_module: str,
) -> tuple[tuple[object, ...], int]:
    seen: set[int] = set()
    nodes: list[object] = []
    byte_count = 0
    stack = [(root, 0) for root in roots]
    while stack:
        value, depth = stack.pop()
        if depth > _OBJECT_GRAPH_MAX_DEPTH:
            raise ObjectGraphLimitError(
                f"plugin-owned object graph exceeded depth limit {_OBJECT_GRAPH_MAX_DEPTH}"
            )
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        nodes.append(value)
        if len(nodes) > _OBJECT_GRAPH_MAX_NODES:
            raise ObjectGraphLimitError(
                f"plugin-owned object graph exceeded node limit {_OBJECT_GRAPH_MAX_NODES}"
            )
        byte_count += _graph_node_bytes(value)
        if byte_count > _OBJECT_GRAPH_MAX_BYTES:
            raise ObjectGraphLimitError(
                f"plugin-owned object graph exceeded byte limit {_OBJECT_GRAPH_MAX_BYTES}"
            )
        children = _object_graph_children(value, plugin_module)
        stack.extend((child, depth + 1) for child in reversed(children))
    return tuple(nodes), byte_count


def _audit_family_storage(
    family: object,
    arguments: tuple[object, ...],
    module_before: Mapping[str, _StorageRootFingerprint],
) -> tuple[bool, bool, tuple[BaseException, ...]]:
    plugin_module = type(family).__module__
    argument_nodes, _ = _walk_object_graph(arguments, plugin_module)
    aliases = {
        id(value)
        for value in argument_nodes
        if not (
            value is None
            or isinstance(value, (str, bytes, bytearray, memoryview, bool, int, float, Enum, type))
            or isinstance(value, (tuple, frozenset)) and not value
        )
    }
    sensitive_payloads = tuple(
        payload
        for argument in arguments
        if isinstance(argument, _CALLBACK_RECORDS)
        for payload in [_semantic_payload(argument, plugin_module)]
        if len(json.dumps(payload, sort_keys=True, separators=(",", ":"))) >= 32
    )
    storage_roots, storage_failures = _family_storage_roots(family, module_before)
    owned_nodes, _ = _walk_object_graph(storage_roots, plugin_module)
    retained = False
    private_rng = False
    for value in owned_nodes:
        if id(value) in aliases:
            retained = True
        if isinstance(value, _PRIVATE_RNG_TYPES):
            private_rng = True
        if sensitive_payloads:
            try:
                candidate = _semantic_payload(value, plugin_module)
            except TypeError:
                continue
            if any(
                _semantic_contains(candidate, sensitive)
                for sensitive in sensitive_payloads
            ):
                retained = True
    return retained, private_rng, storage_failures


def _detached_callback_value(value: object, *, seen: set[int] | None = None) -> object:
    """Convert callback data to a deterministic detached comparison value."""

    if value is None or isinstance(value, (str, bool)) or type(value) is int:
        return value
    if type(value) is float:
        return ("float", repr(value))
    if isinstance(value, Enum):
        return (type(value).__qualname__, value.value)
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return ("cycle", type(value).__module__, type(value).__qualname__)
    seen.add(identity)
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _detached_callback_value(item, seen=seen))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_detached_callback_value(item, seen=seen) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(
            (_detached_callback_value(item, seen=seen) for item in value),
            key=repr,
        ))
    if isinstance(value, np.ndarray):
        return (str(value.dtype), tuple(value.shape), value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (field.name, _detached_callback_value(getattr(value, field.name), seen=seen))
                for field in fields(value)
            ),
        )
    return (type(value).__module__, type(value).__qualname__, repr(value))


class _CallbackProbeFamily(LawFamily):
    """Transparent family proxy that audits every callback boundary."""

    def __init__(self, family: LawFamily, fail: Callable[[str, BaseException, int | None], None],
                 seed: int | None) -> None:
        self._family = family
        self._fail = fail
        self._seed = seed

    @property
    def descriptor(self) -> FamilyDescriptor:
        return self._family.descriptor

    def _invoke(self, name: str, arguments: tuple[object, ...], callback: Callable[[], Any]) -> Any:
        before = _detached_callback_value(arguments)
        module_before = _module_namespace_snapshot(self._family)
        for root_name, root_state in module_before.items():
            if root_state.failure is not None:
                self._fail("callback_isolation", root_state.failure, self._seed)
            if root_state.private_rng:
                self._fail(
                    "ambient_rng",
                    RuntimeError(
                        f"module global {root_name!r} contains a private Python or NumPy RNG object"
                    ),
                    self._seed,
                )
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            return callback()
        finally:
            try:
                after = _detached_callback_value(arguments)
                if after != before:
                    self._fail(
                        "callback_isolation",
                        RuntimeError(f"{name} mutated a detached callback input"),
                        self._seed,
                    )
            except Exception as exc:
                self._fail("callback_isolation", exc, self._seed)
            changed = (
                random.getstate() != python_state
                or not _numpy_state_equal(np.random.get_state(), numpy_state)
            )
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            if changed:
                self._fail(
                    "ambient_rng",
                    RuntimeError(f"{name} changed ambient Python or NumPy RNG state"),
                    self._seed,
                )
            try:
                retained, private_rng, storage_failures = _audit_family_storage(
                    self._family, arguments, module_before,
                )
            except Exception as exc:
                self._fail("callback_isolation", exc, self._seed)
            else:
                if retained:
                    self._fail(
                        "callback_isolation",
                        RuntimeError(f"{name} retained callback input data"),
                        self._seed,
                    )
                if private_rng:
                    self._fail(
                        "ambient_rng",
                        RuntimeError(f"{name} retained a private Python or NumPy RNG object"),
                        self._seed,
                    )
                for failure in storage_failures:
                    self._fail("callback_isolation", failure, self._seed)

    def sample(self, context: SampleContext) -> FamilyInstance:
        return self._invoke("sample", (context,), lambda: self._family.sample(context))

    def channels(self, instance: FamilyInstance, config: Any) -> tuple[ChannelSpec, ...]:
        return self._invoke(
            "channels", (instance, config), lambda: self._family.channels(instance, config),
        )

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        return self._invoke("derive", (view, instance), lambda: self._family.derive(view, instance))

    def apply_proposal(self, proposal: ProposalDraw, view: SubstrateView,
                       instance: FamilyInstance,
                       derived: DerivedLawState) -> LawTransition | None:
        return self._invoke(
            "apply_proposal", (proposal, view, instance, derived),
            lambda: self._family.apply_proposal(proposal, view, instance, derived),
        )

    def filter_kernel_proposal(self, proposal: ProposalDraw, view: SubstrateView,
                               instance: FamilyInstance,
                               derived: DerivedLawState) -> KernelProposalRejection | None:
        return self._invoke(
            "filter_kernel_proposal", (proposal, view, instance, derived),
            lambda: self._family.filter_kernel_proposal(proposal, view, instance, derived),
        )

    def synchronize_private_state(self, view: SubstrateView,
                                  instance: FamilyInstance) -> PrivateStateTransition | None:
        return self._invoke(
            "synchronize_private_state", (view, instance),
            lambda: self._family.synchronize_private_state(view, instance),
        )

    def internal_deadline(self, view: SubstrateView, instance: FamilyInstance,
                          derived: DerivedLawState) -> float | None:
        return self._invoke(
            "internal_deadline", (view, instance, derived),
            lambda: self._family.internal_deadline(view, instance, derived),
        )

    def project_public(self, view: PublicSubstrateView, instance: FamilyInstance,
                       derived: DerivedLawState) -> Mapping[str, object]:
        return self._invoke(
            "project_public", (view, instance, derived),
            lambda: self._family.project_public(view, instance, derived),
        )

    def controls(self, instance: FamilyInstance) -> ControlSuite:
        return self._invoke("controls", (instance,), lambda: self._family.controls(instance))

    def intervene(self, control: ControlKind, view: SubstrateView,
                  instance: FamilyInstance) -> InterventionTransition:
        return self._invoke(
            "intervene", (control, view, instance),
            lambda: self._family.intervene(control, view, instance),
        )

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        return self._invoke("evaluate", (trace,), lambda: self._family.evaluate(trace))

    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return self._invoke(
            "calibration_cases", (), lambda: self._family.calibration_cases(),
        )


def _probed_registration(
    registered: RegisteredFamily,
    fail: Callable[[str, BaseException, int | None], None],
    seed: int | None,
) -> RegisteredFamily:
    return RegisteredFamily(
        _CallbackProbeFamily(registered.family, fail, seed),
        registered.origin,
        registered.official,
        registered.fingerprint,
    )


def _forbidden_observation_path(value: object, path: str = "law_observation") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_OBSERVATION_TERMS:
                return f"{path}.{key}"
            nested = _forbidden_observation_path(item, f"{path}.{key}")
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested = _forbidden_observation_path(item, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _hidden_identity_values(value: object) -> tuple[object, ...]:
    result: list[object] = []
    normalized = _json_value(value)
    if isinstance(normalized, str) or (
        isinstance(normalized, (dict, list)) and bool(normalized)
    ):
        result.append(normalized)
    if isinstance(value, Mapping):
        for item in value.values():
            result.extend(_hidden_identity_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.extend(_hidden_identity_values(item))
    return tuple(result)


def _forbidden_observation_value_path(
    value: object, forbidden: tuple[object, ...], path: str = "law_observation",
) -> str | None:
    normalized = _json_value(value)
    if any(normalized == candidate for candidate in forbidden):
        return path
    if isinstance(value, Mapping):
        for key, item in value.items():
            nested = _forbidden_observation_value_path(
                item, forbidden, f"{path}.{key}",
            )
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested = _forbidden_observation_value_path(
                item, forbidden, f"{path}[{index}]",
            )
            if nested is not None:
                return nested
    return None


class _WaitPolicy:
    name = "family-test-kit"

    def decide(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return {"action": {"type": "WAIT", "duration": 0.1}, "memory": ""}


class FamilyTestKit:
    """Validate one exact registered family without network or model calls."""

    def __init__(self, registry: LawRegistry) -> None:
        if not isinstance(registry, LawRegistry):
            raise TypeError("registry must be a LawRegistry")
        self.registry = registry

    def validate(
        self,
        exact_id: str,
        *,
        seeds: Iterable[int] = range(16),
        include_calibration: bool = True,
    ) -> dict[str, Any]:
        validation_python_state = random.getstate()
        validation_numpy_state = np.random.get_state()
        seed_values = tuple(seeds)
        if not seed_values or any(type(seed) is not int or seed < 0 for seed in seed_values):
            raise ValueError("seeds must be a nonempty iterable of nonnegative integers")
        if len(set(seed_values)) != len(seed_values):
            raise ValueError("seeds must not contain duplicates")
        if type(include_calibration) is not bool:
            raise TypeError("include_calibration must be a boolean")

        registered = self.registry.resolve(exact_id)
        family = registered.family
        from worldzero.kernel import Config, FamilyCallbackError, World
        failures: list[dict[str, Any]] = []
        failure_keys: set[tuple[str, int | None, str, str]] = set()
        checks: dict[str, list[dict[str, Any]]] = {
            name: []
            for name in (
                "accounting",
                "ambient_rng",
                "calibration",
                "callback_isolation",
                "controls",
                "descriptor",
                "determinism",
                "lifecycle",
                "observation_boundary",
                "snapshot_replay",
                "state_independent_channels",
            )
        }

        def fail(check: str, exc: BaseException, seed: int | None = None) -> None:
            key = (check, seed, type(exc).__name__, _bounded_message(exc))
            if key in failure_keys:
                return
            failure_keys.add(key)
            row = {
                "check": check,
                "seed": seed,
                "error_type": type(exc).__name__,
                "message": key[3],
            }
            checks[check].append(row)
            failures.append(row)

        def guarded(check: str, callback: Callable[[], Any], seed: int | None = None) -> Any:
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            try:
                value = callback()
            except Exception as exc:
                fail(check, exc, seed)
                if isinstance(exc, FamilyCallbackError) and check != "callback_isolation":
                    fail("callback_isolation", exc, seed)
                return None
            finally:
                changed = (
                    random.getstate() != python_state
                    or not _numpy_state_equal(np.random.get_state(), numpy_state)
                )
                random.setstate(python_state)
                np.random.set_state(numpy_state)
            if changed:
                fail(
                    "ambient_rng",
                    RuntimeError("plugin callback changed ambient Python or NumPy RNG state"),
                    seed,
                )
            return value

        try:
            descriptor = family.descriptor.persistence_dict()
            json.loads(json.dumps(descriptor, sort_keys=True, allow_nan=False))
        except Exception as exc:
            fail("descriptor", exc)

        config = Config(max_decisions=1)
        worlds: dict[int, World] = {}
        for seed in seed_values:
            probed = _probed_registration(registered, fail, seed)
            first = guarded(
                "determinism",
                lambda seed=seed, probed=probed: World(
                    seed, config, family=probed, record=True,
                ),
                seed,
            )
            second = guarded(
                "determinism",
                lambda seed=seed, probed=probed: World(
                    seed, config, family=probed, record=True,
                ),
                seed,
            )
            if not isinstance(first, World) or not isinstance(second, World):
                continue
            worlds[seed] = first
            try:
                if canonical(first.snapshot()) != canonical(second.snapshot()):
                    raise AssertionError("fixed seed produced different initial snapshots")
            except Exception as exc:
                fail("determinism", exc, seed)

            try:
                before = validate_channel_specs(
                    first._family.channels(first._family_instance, config)
                )
                first._family.derive(first._substrate_view(), first._family_instance)
                after = validate_channel_specs(
                    first._family.channels(first._family_instance, config)
                )
                if before != after:
                    raise AssertionError("channel envelope changed after observing substrate state")
            except Exception as exc:
                fail("state_independent_channels", exc, seed)

            self._check_observation(first, registered, fail, seed)
            self._check_controls(first, fail, seed)
            self._check_accounting(first, fail, seed)
            self._check_lifecycle(probed, seed, fail)
            self._check_replay(probed, seed, fail)

        calibration_family = _probed_registration(registered, fail, None).family
        if include_calibration:
            def calibration_check() -> None:
                first = calibration_family.calibration_cases()
                second = calibration_family.calibration_cases()
                if type(first) is not tuple or not first:
                    raise ValueError("calibration_cases() must return a nonempty tuple")
                if canonical([
                    {
                        "case_id": case.case_id,
                        "kind": case.kind,
                        "expected": thaw_json(case.expected),
                        "absolute_tolerance": case.absolute_tolerance,
                        "relative_tolerance": case.relative_tolerance,
                        "samples": case.samples,
                        "parameters": thaw_json(case.parameters),
                    }
                    for case in first
                ]) != canonical([
                    {
                        "case_id": case.case_id,
                        "kind": case.kind,
                        "expected": thaw_json(case.expected),
                        "absolute_tolerance": case.absolute_tolerance,
                        "relative_tolerance": case.relative_tolerance,
                        "samples": case.samples,
                        "parameters": thaw_json(case.parameters),
                    }
                    for case in second
                ]):
                    raise AssertionError("calibration declaration is nondeterministic")
                calibration_suite_fingerprint(calibration_family)
                if registered.official:
                    from worldzero.mathcheck import check_laws

                    result = check_laws(32, families=(calibration_family,))
                    if not result["passed"]:
                        raise AssertionError("benchmark-owned calibration failed")
                else:
                    self._validate_community_calibration(first, checks)

            guarded("calibration", calibration_check)

        check_rows = [
            {"name": name, "passed": not rows, "failures": rows}
            for name, rows in sorted(checks.items())
        ]
        try:
            calibration_sha256: str | None = guarded(
                "calibration",
                lambda: calibration_suite_fingerprint(calibration_family),
            )
            if not isinstance(calibration_sha256, str):
                calibration_sha256 = None
        except Exception as exc:
            calibration_sha256 = None
            if not checks["calibration"]:
                fail("calibration", exc)
            check_rows = [
                {"name": name, "passed": not rows, "failures": rows}
                for name, rows in sorted(checks.items())
            ]
        report = {
            "schema": REPORT_SCHEMA,
            "family_id": exact_id,
            "descriptor": family.descriptor.persistence_dict(),
            "fingerprint": registered.fingerprint,
            "calibration_suite_sha256": calibration_sha256,
            "origin": registered.origin,
            "official": registered.official,
            "experimental": not registered.official,
            "seed_count": len(seed_values),
            "checks": check_rows,
            "failures": failures,
            "passed": not failures,
        }
        random.setstate(validation_python_state)
        np.random.set_state(validation_numpy_state)
        return json.loads(json.dumps(report, sort_keys=True, allow_nan=False))

    @staticmethod
    def _validate_community_calibration(
        cases: tuple[CalibrationCase, ...],
        checks: Mapping[str, list[dict[str, Any]]],
    ) -> None:
        if sum(case.samples for case in cases) > _MAX_COMMUNITY_CALIBRATION_SAMPLES:
            raise ValueError("community calibration sample budget exceeds the bounded maximum")
        for case in cases:
            if not isinstance(case, CalibrationCase):
                raise TypeError("community calibration suite contains an invalid case")
            if case.kind != "validator_contract":
                raise ValueError(
                    f"community calibration case {case.case_id!r} uses an unknown contract kind"
                )
            parameters = thaw_json(case.parameters)
            if not isinstance(parameters, dict) or set(parameters) != {"contract"}:
                raise ValueError(
                    f"community calibration case {case.case_id!r} has invalid contract parameters"
                )
            contract = parameters["contract"]
            if contract not in _COMMUNITY_CALIBRATION_CONTRACTS:
                raise ValueError(
                    f"community calibration case {case.case_id!r} names an unknown contract"
                )
            if (
                thaw_json(case.expected) is not True
                or case.absolute_tolerance != 0.0
                or case.relative_tolerance != 0.0
            ):
                raise ValueError(
                    f"community calibration case {case.case_id!r} must assert exact true"
                )
            observed = not any(
                checks[check_name]
                for check_name in _COMMUNITY_CALIBRATION_CONTRACTS[contract]
            )
            if not observed:
                raise AssertionError(
                    f"community calibration case {case.case_id!r} failed its benchmark contract"
                )

    def _check_observation(
        self, world: "World", registered: RegisteredFamily,
        fail: Callable[[str, BaseException, int | None], None], seed: int,
    ) -> None:
        family = world._family
        try:
            observation = json.loads(json.dumps(world.observe(), allow_nan=False))
            law_observation = observation.get("law_observation", {})
            forbidden = _forbidden_observation_path(law_observation)
            if forbidden is not None:
                raise ValueError(f"forbidden evaluator/hidden observation field: {forbidden}")
            descriptor = registered.family.descriptor
            forbidden_values = (
                descriptor.family_id,
                descriptor.family_version,
                registered.fingerprint,
                calibration_suite_fingerprint(world._family),
                registered.origin,
                *_hidden_identity_values(world._family_instance.hidden_parameters),
                *_hidden_identity_values(world._family_instance.private_state),
            )
            leaked = _forbidden_observation_value_path(
                law_observation, forbidden_values,
            )
            if leaked is not None:
                raise ValueError(
                    f"forbidden evaluator/hidden observation value: {leaked}"
                )

            public_view = world._public_view()
            substrate_view = world._substrate_view()
            derived = family.derive(substrate_view, world._family_instance)
            active = family.project_public(public_view, world._family_instance, derived)
            def perturbations(value: object) -> tuple[object, ...]:
                if type(value) is str:
                    return (f"{value}-worldzero-probe",)
                if type(value) is bool:
                    return (not value,)
                if type(value) is int:
                    return tuple(dict.fromkeys((value + 1, max(0, value - 1))))
                if type(value) is float:
                    return tuple(dict.fromkeys((value + 1.0, max(0.0, value - 1.0))))
                if isinstance(value, tuple):
                    candidates: list[object] = []
                    reversed_value = tuple(reversed(value))
                    if reversed_value != value:
                        candidates.append(reversed_value)
                    for index, item in enumerate(value):
                        for replacement in perturbations(item):
                            candidate = list(value)
                            candidate[index] = replacement
                            candidates.append(tuple(candidate))
                    return tuple(candidates)
                if isinstance(value, Mapping):
                    candidates = []
                    for key, item in value.items():
                        for replacement in perturbations(item):
                            candidate = dict(value)
                            candidate[key] = replacement
                            candidates.append(candidate)
                    return tuple(candidates)
                return ()

            original = world._family_instance
            variants = [
                FamilyInstance(
                    "worldzero_probe:identity", "9.9.9",
                    original.hidden_parameters, original.private_state,
                    not original.enabled,
                ),
            ]
            for key, value in original.hidden_parameters.items():
                for replacement in perturbations(value):
                    hidden = dict(original.hidden_parameters)
                    hidden[key] = replacement
                    variants.append(FamilyInstance(
                        original.family_id, original.family_version,
                        hidden, original.private_state, original.enabled,
                    ))
            for key, value in original.private_state.items():
                for replacement in perturbations(value):
                    private = dict(original.private_state)
                    private[key] = replacement
                    variants.append(FamilyInstance(
                        original.family_id, original.family_version,
                        original.hidden_parameters, private, original.enabled,
                    ))

            derived_variants = 0
            for variant in variants:
                try:
                    variant_derived = family.derive(substrate_view, variant)
                    variant_projection = family.project_public(
                        public_view, variant, variant_derived,
                    )
                except Exception:
                    continue
                derived_variants += 1
                if canonical(active) != canonical(variant_projection):
                    raise ValueError(
                        "public projection depends on hidden/private/control state"
                    )
            if derived_variants == 0:
                raise ValueError("no bounded hidden-state projection probe was derivable")

            probe = copy.deepcopy(observation)
            probe["position"] = [-1, -1]
            if world.observe()["position"] == [-1, -1]:
                raise AssertionError("policy observation aliases kernel state")
        except Exception as exc:
            fail("observation_boundary", exc, seed)

    def _check_controls(
        self, world: "World", fail: Callable[[str, BaseException, int | None], None], seed: int,
    ) -> None:
        try:
            controls = world._validated_controls()
            for kind in ControlKind:
                spec = getattr(controls, kind.value)
                if spec.matching_constraints != _REQUIRED_MATCHING_CONSTRAINTS:
                    raise ValueError(
                        f"{kind.value} control must declare the exact benchmark matching constraints"
                    )
            def check_disabled(base: "World", kind: ControlKind) -> None:
                base_material = base.material_count()
                base_modules = copy.deepcopy(base.modules)
                base_resources = base.resources.copy()
                base_observation = base.observe()
                base_pending = copy.deepcopy(base._pending)
                base_rng = copy.deepcopy(base.rng.bit_generator.state)
                branch = base.clone(record=True)
                branch.apply_control(kind)
                if branch._family_instance.enabled:
                    raise AssertionError(f"{kind.value} did not disable the mechanism")
                if branch.material_count() != base_material:
                    raise AssertionError(f"{kind.value} changed material")
                if branch.modules != base_modules or not np.array_equal(
                    branch.resources, base_resources,
                ):
                    raise AssertionError(f"{kind.value} changed public substrate")
                if branch.observe() != base_observation:
                    raise AssertionError(f"{kind.value} changed matched public observation")
                if branch._pending != base_pending or branch.rng.bit_generator.state != base_rng:
                    raise AssertionError(f"{kind.value} changed matched proposal state")
                error = branch.accounting_error()
                if abs(error["energy"]) > 1e-7 or error["material"] != 0:
                    raise AssertionError(f"{kind.value} violated accounting")

            def check_retained(base: "World") -> None:
                base_material = base.material_count()
                base_modules = copy.deepcopy(base.modules)
                base_resources = base.resources.copy()
                base_observation = base.observe()
                branch = base.clone(record=True)
                retained_pending = copy.deepcopy(branch._pending)
                retained_rng = copy.deepcopy(branch.rng.bit_generator.state)
                retained = branch._family.intervene(
                    ControlKind.RETAINED,
                    branch._substrate_view(),
                    branch._family_instance,
                )
                if (
                    not isinstance(retained, InterventionTransition)
                    or retained.control is not ControlKind.RETAINED
                    or retained.resulting_instance != branch._family_instance
                    or retained.operations
                    or retained.declared_capabilities
                    or retained.accounting != AccountingDelta()
                ):
                    raise AssertionError("retained control changed the family or substrate")
                branch._apply_family_transition(retained)
                if (
                    branch.material_count() != base_material
                    or branch.observe() != base_observation
                    or branch.modules != base_modules
                    or not np.array_equal(branch.resources, base_resources)
                    or branch._pending != retained_pending
                    or branch.rng.bit_generator.state != retained_rng
                ):
                    raise AssertionError(
                        "retained control violated matched public/proposal state"
                    )
                error = branch.accounting_error()
                if abs(error["energy"]) > 1e-7 or error["material"] != 0:
                    raise AssertionError("retained control violated accounting")

            def check_broken(base: "World", *, structural: bool) -> None:
                base_material = base.material_count()
                base_resources = base.resources.copy()
                branch = base.clone(record=True)
                broken_pending = copy.deepcopy(branch._pending)
                broken_rng = copy.deepcopy(branch.rng.bit_generator.state)
                if structural:
                    if branch.break_geometry() is not True:
                        raise AssertionError(
                            "broken control was not applied to a structural layout"
                        )
                else:
                    broken = branch._family.intervene(
                        ControlKind.BROKEN,
                        branch._substrate_view(),
                        branch._family_instance,
                    )
                    if (
                        not isinstance(broken, InterventionTransition)
                        or broken.control is not ControlKind.BROKEN
                        or broken.resulting_instance != branch._family_instance
                        or broken.accounting != AccountingDelta()
                    ):
                        raise AssertionError(
                            "broken control changed family identity/accounting"
                        )
                    changes = [
                        operation for operation in broken.operations
                        if isinstance(operation, ModulePositionChange)
                    ]
                    if broken.operations and (
                        len(broken.operations) != 1 or len(changes) != 1
                    ):
                        raise AssertionError("broken control must move exactly one module")
                    branch._apply_family_transition(broken)
                if (
                    branch.material_count() != base_material
                    or not np.array_equal(branch.resources, base_resources)
                    or branch._pending != broken_pending
                    or branch.rng.bit_generator.state != broken_rng
                ):
                    raise AssertionError(
                        "broken control violated matched material/proposal state"
                    )
                error = branch.accounting_error()
                if abs(error["energy"]) > 1e-7 or error["material"] != 0:
                    raise AssertionError("broken control violated accounting")

            def check_all_controls(base: "World", *, structural: bool) -> None:
                check_disabled(base, ControlKind.NULL)
                check_disabled(base, ControlKind.KNOCKOUT)
                check_retained(base)
                check_broken(base, structural=structural)

            initial_structural = (
                world._derived_family_state.state.get("structural") is True
            )
            check_all_controls(world, structural=initial_structural)

            # Exercise row, column, and both diagonal orientations.  Permuting
            # module identity across each bounded layout covers pair-specific laws.
            layout_shapes = (
                ((0, 0), (0, 1), (0, 2)),
                ((0, 0), (1, 0), (2, 0)),
                ((0, 0), (1, 1), (2, 2)),
                ((0, 2), (1, 1), (2, 0)),
            )
            layouts = (
                layout
                for positions in layout_shapes
                for layout in permutations(positions)
            )
            for layout in layouts:
                candidate = world.clone(record=True)
                candidate.modules = list(layout)
                candidate._update_field()
                if candidate._derived_family_state.state.get("structural") is not True:
                    continue
                check_all_controls(candidate, structural=True)
        except Exception as exc:
            fail("controls", exc, seed)

    def _check_accounting(
        self, world: "World", fail: Callable[[str, BaseException, int | None], None], seed: int,
    ) -> None:
        try:
            for channel in world._family_channels:
                domain_size = (
                    len(world.modules)
                    if channel.target_domain is TargetDomain.MODULE
                    else world.config.width * world.config.height
                    if channel.target_domain is TargetDomain.CELL
                    else 1
                )
                targets = range(domain_size)
                uniforms = (
                    (0.0, 0.5, float(np.nextafter(1.0, 0.0)))
                    if DrawRequirement.ACCEPTANCE_UNIFORM in channel.draw_requirements
                    else (0.0,)
                )
                for target in targets:
                    for uniform in uniforms:
                        branch = world.clone(record=True)
                        branch.normalize_resources(np.ones_like(branch.resources))
                        branch._proposal(channel.channel_id, target, uniform)
                        error = branch.accounting_error()
                        if abs(error["energy"]) > 1e-7 or error["material"] != 0:
                            raise AssertionError(
                                "family transition violated kernel accounting"
                            )
            cell_count = world.config.width * world.config.height
            for target in range(cell_count):
                for uniform in (0.0, 0.5, float(np.nextafter(1.0, 0.0))):
                    branch = world.clone(record=True)
                    branch.normalize_resources(np.ones_like(branch.resources))
                    branch._proposal("raw_decay", target, uniform)
                    error = branch.accounting_error()
                    if abs(error["energy"]) > 1e-7 or error["material"] != 0:
                        raise AssertionError("kernel proposal filter violated accounting")
        except Exception as exc:
            fail("accounting", exc, seed)

    def _check_lifecycle(
        self, registered: Any, seed: int,
        fail: Callable[[str, BaseException, int | None], None],
    ) -> None:
        try:
            from worldzero.kernel import Config, World

            config = Config(lifespan=0.1, max_decisions=1)
            world = World(seed, config, family=registered, record=True)
            world.step({"action": {"type": "WAIT", "duration": 0.1}, "memory": "private"})
            if world.agent is None or world.agent.alive:
                raise AssertionError("death is not absorbing at the lifespan boundary")
            if world.agent.memory or world.agent.last_result or world.agent.inventory is not None:
                raise AssertionError("death did not clear private successor state")
            try:
                world.step({"action": {"type": "WAIT", "duration": 0.1}, "memory": ""})
            except RuntimeError:
                pass
            else:
                raise AssertionError("dead agent accepted a later action")
            world.retire()
            world.spawn(2)
            if world.agent is None or world.agent.memory or world.agent.last_result:
                raise AssertionError("successor did not start with empty private memory")
        except Exception as exc:
            fail("lifecycle", exc, seed)

    def _check_replay(
        self, registered: Any, seed: int,
        fail: Callable[[str, BaseException, int | None], None],
    ) -> None:
        try:
            from worldzero.experiment import run_episode, verify_plugin_replay
            from worldzero.kernel import Config, World

            world = World(seed, Config(max_decisions=1), family=registered, record=True)
            _, trace = run_episode(world, _WaitPolicy(), capture=True)
            if trace is None:
                raise AssertionError("trace-v4 capture returned no trace")
            result = verify_plugin_replay(trace, registry=self.registry)
            if result.get("verified") is not True:
                raise AssertionError("trace-v4 replay did not verify")
            restored = World.from_snapshot(trace["final"], registry=self.registry)
            if canonical(restored.snapshot()) != canonical(trace["final"]):
                raise AssertionError("state-v3 snapshot roundtrip drifted")
        except Exception as exc:
            fail("snapshot_replay", exc, seed)


__all__ = ["FamilyTestKit", "REPORT_SCHEMA"]
