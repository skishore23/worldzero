"""Post-final-review mapping-fingerprint hardening regressions."""

from __future__ import annotations

import random
import sys

from worldzero.laws import EvaluatorTrace, FamilyEvidence, thaw_json
from worldzero.laws.testing import (
    ObjectGraphLimitError,
    ObjectGraphUnsupportedError,
    _root_fingerprint_changed,
    _StorageRootFingerprint,
    _storage_content_fingerprint,
)

from test_task6_fix_round2 import ConformingCommunityFamily, _assert_rejected, _validate


_ROOT_NAME = "_worldzero_typed_mapping_root"


class LocalConformingFamily(ConformingCommunityFamily):
    """Conforming control implemented in this callback module."""


def _detached_trace(trace: EvaluatorTrace) -> dict[str, object]:
    return {
        "events": [thaw_json(event) for event in trace.events],
        "terminal": thaw_json(trace.terminal),
    }


class MixedIntStringTraceFamily(ConformingCommunityFamily):
    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        root = vars(sys.modules[type(self).__module__])[_ROOT_NAME]
        root[1] = _detached_trace(trace)
        return super().evaluate(trace)


def test_mixed_integer_and_string_key_collision_cannot_hide_detached_trace() -> None:
    module = sys.modules[MixedIntStringTraceFamily.__module__]
    vars(module)[_ROOT_NAME] = {1: None, "1": "unchanged-visible-value"}
    try:
        _assert_rejected(MixedIntStringTraceFamily(), "callback_isolation")
    finally:
        vars(module).pop(_ROOT_NAME, None)


class _SameStringKey:
    def __init__(self, label: str) -> None:
        self.label = label

    def __str__(self) -> str:
        return "same-string"


class SameStringCustomKeyTraceFamily(ConformingCommunityFamily):
    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        root = vars(sys.modules[type(self).__module__])[_ROOT_NAME]
        target = next(
            key for key in dict.__iter__(root)
            if isinstance(key, _SameStringKey) and key.label == "target"
        )
        root[target] = _detached_trace(trace)
        return super().evaluate(trace)


def test_custom_keys_with_identical_string_forms_remain_distinct() -> None:
    module = sys.modules[SameStringCustomKeyTraceFamily.__module__]
    vars(module)[_ROOT_NAME] = {
        _SameStringKey("target"): None,
        _SameStringKey("visible"): "unchanged-visible-value",
    }
    try:
        _assert_rejected(SameStringCustomKeyTraceFamily(), "callback_isolation")
    finally:
        vars(module).pop(_ROOT_NAME, None)


class MixedKeyNestedRngFamily(ConformingCommunityFamily):
    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        root = vars(sys.modules[type(self).__module__])[_ROOT_NAME]
        root[1] = {"nested": [random.Random(71)]}
        return super().evaluate(trace)


def test_private_rng_flag_drift_is_a_changed_root_even_if_hash_collides() -> None:
    module = sys.modules[MixedKeyNestedRngFamily.__module__]
    vars(module)[_ROOT_NAME] = {1: None, "1": "unchanged-visible-value"}
    try:
        _assert_rejected(MixedKeyNestedRngFamily(), "ambient_rng")
    finally:
        vars(module).pop(_ROOT_NAME, None)


def _fingerprint(value: object) -> tuple[str, bool]:
    return _storage_content_fingerprint(value, __name__)


def test_value_key_types_have_distinct_internal_fingerprints() -> None:
    assert _fingerprint({True: "value"}) != _fingerprint({1: "value"})
    assert _fingerprint({1: "value"}) != _fingerprint({"1": "value"})
    assert _fingerprint({b"1": "value"}) != _fingerprint({"1": "value"})
    assert _fingerprint({(1, "1"): "value"}) != _fingerprint({frozenset({1, "1"}): "value"})


def test_swapped_mixed_key_branches_have_distinct_fingerprints() -> None:
    left = {1: "left", "1": "right"}
    right = {1: "right", "1": "left"}
    assert _fingerprint(left) != _fingerprint(right)


def test_non_hash_state_flags_are_part_of_root_drift() -> None:
    base = _StorageRootFingerprint(7, "same", False)
    assert _root_fingerprint_changed(
        base, _StorageRootFingerprint(7, "same", True),
    )
    assert _root_fingerprint_changed(
        base,
        _StorageRootFingerprint(
            7, "same", False, ObjectGraphLimitError("bounded overflow"),
        ),
    )
    assert _root_fingerprint_changed(
        base,
        _StorageRootFingerprint(
            7, "same", False, ObjectGraphUnsupportedError("unsupported root"),
        ),
    )


def test_unchanged_mixed_and_custom_key_roots_remain_allowed() -> None:
    module = sys.modules[LocalConformingFamily.__module__]
    roots = (
        {1: "integer", "1": "string"},
        {_SameStringKey("left"): "left", _SameStringKey("right"): "right"},
    )
    for root in roots:
        vars(module)[_ROOT_NAME] = root
        try:
            report = _validate(LocalConformingFamily())
        finally:
            vars(module).pop(_ROOT_NAME, None)
        assert report["passed"] is True, report
