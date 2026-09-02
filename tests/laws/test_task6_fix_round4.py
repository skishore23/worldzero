"""Fourth-round module-storage provenance regressions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import random
import sys
from typing import Callable

import pytest

from worldzero.laws import EvaluatorTrace, FamilyEvidence, SampleContext, thaw_json

from test_task6_fix_round2 import (
    ConformingCommunityFamily,
    _assert_rejected,
    _validate,
)


_MODULE_ROOT_NAME = "_worldzero_preexisting_mutable_root"


class _CustomRoot:
    def __init__(self, payload: object = None) -> None:
        self.payload = payload


class _SlottedRoot:
    __slots__ = ("payload",)

    def __init__(self, payload: object = None) -> None:
        self.payload = payload


class LocalConformingFamily(ConformingCommunityFamily):
    """Conforming control implemented in this callback module."""


_MAPPING_PROPERTY_READS = 0


class _PropertyMapping(Mapping[str, object]):
    def __init__(self) -> None:
        self.payload = {"ordinary": True}

    def __getitem__(self, key: str) -> object:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    @property
    def items(self) -> object:
        global _MAPPING_PROPERTY_READS
        _MAPPING_PROPERTY_READS += 1
        raise AssertionError("custom mapping property executed")


def _store_detached_trace(root: object, trace: EvaluatorTrace) -> None:
    payload = {
        "events": [thaw_json(event) for event in trace.events],
        "terminal": thaw_json(trace.terminal),
    }
    if isinstance(root, dict):
        root["payload"] = payload
    elif isinstance(root, list):
        root.append(payload)
    else:
        descriptor = vars(type(root)).get("payload")
        if hasattr(type(root), "__slots__") and descriptor is not None:
            descriptor.__set__(root, payload)
        else:
            vars(root)["payload"] = payload


class InPlaceModuleTraceFamily(ConformingCommunityFamily):
    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        module = sys.modules[type(self).__module__]
        _store_detached_trace(vars(module)[_MODULE_ROOT_NAME], trace)
        return super().evaluate(trace)


@pytest.mark.parametrize(
    "factory",
    (
        pytest.param(dict, id="dict"),
        pytest.param(list, id="list"),
        pytest.param(_CustomRoot, id="custom-dict"),
        pytest.param(_SlottedRoot, id="custom-slots"),
    ),
)
def test_in_place_mutated_preexisting_module_roots_are_rejected(
    factory: Callable[[], object],
) -> None:
    module = sys.modules[InPlaceModuleTraceFamily.__module__]
    vars(module)[_MODULE_ROOT_NAME] = factory()
    try:
        _assert_rejected(InPlaceModuleTraceFamily(), "callback_isolation")
    finally:
        vars(module).pop(_MODULE_ROOT_NAME, None)


class InPlaceModuleRngFamily(ConformingCommunityFamily):
    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        module = sys.modules[type(self).__module__]
        root = vars(module)[_MODULE_ROOT_NAME]
        root["nested"].append({"private_rng": random.Random(17)})
        return super().evaluate(trace)


def test_nested_rng_in_in_place_mutated_module_root_is_rejected() -> None:
    module = sys.modules[InPlaceModuleRngFamily.__module__]
    vars(module)[_MODULE_ROOT_NAME] = {"nested": []}
    try:
        _assert_rejected(InPlaceModuleRngFamily(), "ambient_rng")
    finally:
        vars(module).pop(_MODULE_ROOT_NAME, None)


@pytest.mark.parametrize(
    "root",
    (
        pytest.param({"ordinary": [1, 2, 3]}, id="dict-list"),
        pytest.param(_CustomRoot({"ordinary": True}), id="custom-dict"),
        pytest.param(_SlottedRoot({"ordinary": True}), id="custom-slots"),
        pytest.param(
            SampleContext({"module_count": 3}, {"law": 7}),
            id="sample-context",
        ),
    ),
)
def test_unchanged_preexisting_module_roots_remain_allowed(root: object) -> None:
    module = sys.modules[LocalConformingFamily.__module__]
    vars(module)[_MODULE_ROOT_NAME] = root
    try:
        report = _validate(LocalConformingFamily())
    finally:
        vars(module).pop(_MODULE_ROOT_NAME, None)
    assert report["passed"] is True, report


def test_module_fingerprinting_never_invokes_custom_mapping_properties() -> None:
    global _MAPPING_PROPERTY_READS
    _MAPPING_PROPERTY_READS = 0
    module = sys.modules[LocalConformingFamily.__module__]
    vars(module)[_MODULE_ROOT_NAME] = _PropertyMapping()
    try:
        report = _validate(LocalConformingFamily())
    finally:
        vars(module).pop(_MODULE_ROOT_NAME, None)
    assert report["passed"] is True, report
    assert _MAPPING_PROPERTY_READS == 0


def test_over_budget_preexisting_mutable_module_root_fails_structurally() -> None:
    module = sys.modules[LocalConformingFamily.__module__]
    vars(module)[_MODULE_ROOT_NAME] = [[index] for index in range(4_097)]
    try:
        report = _assert_rejected(LocalConformingFamily(), "callback_isolation")
    finally:
        vars(module).pop(_MODULE_ROOT_NAME, None)
    failures = [
        row for row in report["failures"]
        if row["check"] == "callback_isolation"
    ]
    assert any(
        row["error_type"] == "ObjectGraphLimitError"
        and "module global" in row["message"]
        and "node limit" in row["message"]
        for row in failures
    ), failures
