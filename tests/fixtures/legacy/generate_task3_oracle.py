"""Reproduce the pre-Task-3 catalysis/null compatibility oracle.

Run only against the legacy ``worldzero.core`` source hash recorded below.
The output is a compatibility fixture, not benchmark or model evidence.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from worldzero.core import Config, Law, RAW, World
from worldzero.util import canonical, digest


FIXTURE_DIR = Path(__file__).parent
ORACLE = FIXTURE_DIR / "task-3-legacy-oracle.json.gz"
SOURCE_SHA256 = "fb4bfc7dafce3ea1d6685238f71f6124e1f80db85f620a32d333162cc5a1b45d"
SEEDS = tuple(range(330_000, 330_064))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(world: World, event_start: int) -> dict[str, object]:
    return {
        "accounting_error": world.accounting_error(),
        "agent": asdict(world.agent) if world.agent is not None else None,
        "assemblies": world.assemblies,
        "audit": copy.deepcopy(world.audit),
        "channels": [[name, rate] for name, rate in world._channels],
        "conversions": world.conversions,
        "conversions_without_living_agent": world.conversions_without_living_agent,
        "event_count": world.event_count,
        "events_since_previous": copy.deepcopy(world.events[event_start:]),
        "field": sorted(world._field),
        "first_assembly": world.first_assembly,
        "functional": world.functional_motif(),
        "history_sha256": world.history_hash,
        "integrated_motif_time": world.integrated_motif_time,
        "law": asdict(world.law),
        "mechanism_enabled": world.mechanism_enabled,
        "modules": [list(position) if position is not None else None for position in world.modules],
        "pending": list(world._pending) if world._pending is not None else None,
        "proposal_count": world.proposal_count,
        "regime": world.regime,
        "resources": world.resources.tolist(),
        "rng": copy.deepcopy(world.rng.bit_generator.state),
        "structural": world.structural_match(),
        "time": world.time,
    }


def _legacy_world(seed: int, family: str, scenario: str) -> World:
    config = Config() if scenario == "ordinary" else replace(Config(), lifespan=0.35)
    world = World(seed, config, record=True)
    world.law = Law(world.law.pair, family, world.law.geometry)
    if scenario == "ordinary":
        first, second = world.law.pair
        third = next(index for index in range(3) if index not in world.law.pair)
        world.modules[first] = world.home
        world.modules[second] = (world.home[0], world.home[1] + 1)
        world.modules[third] = (0, 0)
        resources = np.zeros_like(world.resources)
        resources[world.home[0] - 1, world.home[1]] = RAW
        world.normalize_resources(resources)
    world._update_field()
    return world


def _trajectory(seed: int, family: str, scenario: str) -> dict[str, object]:
    world = _legacy_world(seed, family, scenario)
    actions = (
        ({"action": {"type": "WAIT", "duration": 8.0}, "memory": "oracle"},) * 4
        if scenario == "ordinary"
        else ({"action": {"type": "WAIT", "duration": 0.1}, "memory": "dies"},) * 3
    )
    boundaries: list[dict[str, object]] = []
    event_start = 0
    boundaries.append(
        {
            "action": None,
            "observation": world.observe(),
            "result": None,
            "state": _state(world, event_start),
        }
    )
    event_start = len(world.events)
    for decision in actions:
        if world.agent is None or not world.agent.alive:
            break
        observation = world.observe()
        result = world.step(copy.deepcopy(decision))
        boundaries.append(
            {
                "action": decision,
                "observation": observation,
                "result": result,
                "state": _state(world, event_start),
            }
        )
        event_start = len(world.events)
    snapshot = world.snapshot()
    return {
        "boundaries": boundaries,
        "family": family,
        "scenario": scenario,
        "seed": seed,
        "terminal_snapshot": snapshot,
        "terminal_snapshot_sha256": digest(snapshot),
    }


def main() -> None:
    root = FIXTURE_DIR.parents[2]
    core = root / "worldzero" / "core.py"
    if _sha256(core) != SOURCE_SHA256:
        raise SystemExit("refusing to generate: worldzero/core.py is not the frozen pre-Task-3 source")
    trajectories = [
        _trajectory(seed, family, scenario)
        for family in ("catalysis", "null")
        for seed in SEEDS
        for scenario in ("ordinary", "short_lifespan")
    ]
    payload = {
        "schema": "worldzero-task-3-legacy-oracle-v1",
        "provenance": {
            "actions": {
                "ordinary": "four WAIT(8.0) decisions after deterministic functional geometry/resource preparation",
                "short_lifespan": "WAIT(0.1) decisions until absorbing death with lifespan=0.35",
            },
            "core_source_sha256": SOURCE_SHA256,
            "policy": "local deterministic actions; no model or endpoint calls",
            "seeds": list(SEEDS),
        },
        "trajectories": trajectories,
    }
    encoded = canonical(payload).encode("utf-8")
    with ORACLE.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)


if __name__ == "__main__":
    main()
