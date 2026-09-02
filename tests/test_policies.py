import copy
import json
from dataclasses import replace

from worldzero.core import Config, World
from worldzero.experiment import run_episode, simulate, verify_replay
from worldzero.policies import BlindManipulatorPolicy


def observation(*, inventory=None, position=(3, 3), objects=None, last_result=None):
    objects = objects if objects is not None else [
        {"id": "module-a", "consume": False, "pick": True},
    ]
    return {
        "time": 0.0,
        "age": 0.0,
        "remaining": 100.0,
        "energy": 22.0,
        "position": list(position),
        "bounds": [8, 8],
        "inventory": inventory,
        "inventory_state": {"occupied": inventory is not None, "object_id": inventory},
        "current_cell": {"position": list(position), "surface": 1, "objects": objects},
        "legal_actions": {
            "MOVE": {"directions": ["N", "E", "S", "W"]},
            "PICK": {"available": inventory is None and bool(objects)},
            "DROP": {"available": inventory is not None},
            "CONSUME": {"available": False},
            "WAIT": {"duration_min": 0.1, "duration_max": 8.0},
        },
        "local": [{"position": list(position), "surface": 1, "objects": objects}],
        "last_result": {} if last_result is None else last_result,
        "memory": "",
    }


def frozen_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_blind_manipulator_confirms_scheduled_cycle_and_never_mutates_observation():
    """Fails if requested PICK/DROP actions count before public confirmation."""
    policy = BlindManipulatorPolicy(73)
    policy.decision_count = 23
    start = observation()
    expected_start = frozen_json(copy.deepcopy(start))

    first = policy.decide(start)

    assert first["action"] == {"type": "PICK"}
    assert policy.manipulation_cycles_started == 0
    assert frozen_json(start) == expected_start

    moves = []
    while True:
        carrying = observation(
            inventory="module-a", objects=[],
            last_result={"action": {"type": "PICK"}, "status": "picked", "object_id": "module-a"},
        )
        expected_carrying = frozen_json(copy.deepcopy(carrying))
        decision = policy.decide(carrying)
        assert frozen_json(carrying) == expected_carrying
        if decision["action"]["type"] == "DROP":
            break
        moves.append(decision["action"])

    assert 2 <= len(moves) <= 8
    assert all(move["type"] == "MOVE" for move in moves)
    assert policy.manipulation_cycles_started == 1
    assert policy.manipulation_cycles_completed == 0

    dropped = observation(
        objects=[],
        last_result={"action": {"type": "DROP"}, "status": "dropped", "object_id": "module-a"},
    )
    policy.decide(dropped)
    assert policy.manipulation_cycles_completed == 1


def test_blind_manipulator_does_not_count_ineffective_pick_or_drop():
    """Fails if a no-effect primitive request consumes a cycle or completes one."""
    policy = BlindManipulatorPolicy(73)
    policy.decision_count = 23
    assert policy.decide(observation())["action"]["type"] == "PICK"

    failed_pick = observation(
        objects=[],
        last_result={"action": {"type": "PICK"}, "status": "no_effect", "object_id": None},
    )
    policy.decide(failed_pick)
    assert policy.manipulation_cycles_started == 0
    assert policy.cycle_index == 0

    policy.phase = "carry"
    policy.target_id = "module-a"
    policy.walk_remaining = 0
    drop = policy.decide(observation(inventory="module-a", objects=[]))
    assert drop["action"]["type"] == "DROP"
    assert policy.manipulation_cycles_completed == 0

    failed_drop = observation(
        inventory="module-a", objects=[],
        last_result={"action": {"type": "DROP"}, "status": "no_effect", "object_id": None},
    )
    policy.decide(failed_drop)
    assert policy.manipulation_cycles_completed == 0


def test_blind_manipulator_does_not_count_cognition_preempted_actions():
    """Fails if cognition/lifespan preemption counts an unexecuted pickup or drop."""
    preempted = replace(Config(), cognition_time=1.0, lifespan=0.1)

    pick_policy = BlindManipulatorPolicy(73)
    pick_policy.decision_count = 23
    pick_world = World(73, preempted)
    pick_world.agent.position = pick_world.modules[0]
    pick = pick_policy.decide(pick_world.observe())
    assert pick["action"]["type"] == "PICK"
    assert pick_world.step(pick)["status"] == "terminated"
    assert not pick_world.agent.alive
    assert pick_policy.manipulation_cycles_started == 0
    assert pick_policy.cycle_index == 0

    drop_policy = BlindManipulatorPolicy(74)
    drop_policy.phase = "carry"
    drop_policy.target_id = "module-a"
    drop_policy.walk_remaining = 0
    drop_world = World(74, preempted)
    drop_world.agent.inventory = 0
    drop_world.modules[0] = None
    drop = drop_policy.decide(drop_world.observe())
    assert drop["action"]["type"] == "DROP"
    assert drop_world.step(drop)["status"] == "terminated"
    assert not drop_world.agent.alive
    assert drop_policy.manipulation_cycles_completed == 0


def test_blind_manipulator_records_successful_final_pick_once():
    """Fails if a final, physical PICK waits for an unavailable next decide call."""
    policy = BlindManipulatorPolicy(73)
    policy.decision_count = 23
    world = World(73, replace(Config(), max_decisions=1))
    world.agent.position = world.modules[0]

    result, _ = run_episode(world, policy)

    assert result["decisions"] == 1
    assert result["manipulation_cycles_started"] == 1
    assert policy.manipulation_cycles_started == 1
    assert policy.cycle_index == 1


def test_blind_manipulator_records_successful_final_drop_once():
    """Fails if a final, physical DROP waits for an unavailable next decide call."""
    policy = BlindManipulatorPolicy(74)
    policy.phase = "carry"
    policy.target_id = "module-a"
    policy.walk_remaining = 0
    policy.cycle_index = 1
    policy.manipulation_cycles_started = 1
    world = World(74, replace(Config(), max_decisions=1))
    world.agent.inventory = 0
    world.modules[0] = None
    policy.target_id = world.symbols[2]

    result, _ = run_episode(world, policy)

    assert result["decisions"] == 1
    assert result["manipulation_cycles_completed"] == 1
    assert policy.manipulation_cycles_completed == 1


def test_blind_manipulator_feedback_reconciliation_is_idempotent():
    """Fails if runner feedback and the next decide call double-count a pickup."""
    policy = BlindManipulatorPolicy(73)
    policy.decision_count = 23
    assert policy.decide(observation())["action"]["type"] == "PICK"
    confirmed = observation(
        inventory="module-a", objects=[],
        last_result={"action": {"type": "PICK"}, "status": "picked", "object_id": "module-a"},
    )
    result = confirmed["last_result"]

    policy.after_step(confirmed, result)
    policy.after_step(confirmed, result)
    assert policy.manipulation_cycles_started == 1
    policy.decide(confirmed)
    assert policy.manipulation_cycles_started == 1


def test_blind_manipulator_seed_replicates_and_breaks_equal_distance_ties():
    """Fails if equivalent-object selection is not deterministic and seed-driven."""
    tied = observation(objects=[
        {"id": "module-a", "consume": False, "pick": True},
        {"id": "module-b", "consume": False, "pick": True},
    ])
    sequence = [tied] + [observation(inventory="module-a", objects=[]) for _ in range(9)]
    left = BlindManipulatorPolicy(73)
    right = BlindManipulatorPolicy(73)
    other = BlindManipulatorPolicy(74)
    for policy in (left, right, other):
        policy.decision_count = 23

    left_actions = [policy_action(left, item) for item in sequence]
    right_actions = [policy_action(right, item) for item in sequence]
    other_actions = [policy_action(other, item) for item in sequence]

    assert left_actions == right_actions
    assert left.walk_remaining == right.walk_remaining
    assert left_actions != other_actions


def policy_action(policy, item):
    frozen = frozen_json(copy.deepcopy(item))
    result = policy.decide(item)
    assert frozen_json(item) == frozen
    return result["action"]


def test_blind_manipulator_engine_run_is_replayable_without_model_calls():
    """Fails if the registered control consumes model calls or loses deterministic replay."""
    _, first, first_trace = simulate(73, "blind-manipulator", capture=True)
    _, second, second_trace = simulate(73, "blind-manipulator", capture=True)

    assert first == second
    assert first_trace == second_trace
    assert first["policy"] == "blind-manipulator"
    assert first["model_calls"] == 0
    assert verify_replay(first_trace)["verified"]
    assert "manipulation_cycles_started" in first
    assert "manipulation_cycles_completed" in first
