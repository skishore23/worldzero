from __future__ import annotations

from worldzero.causal_evidence import discriminating_reconstruction


def event(kind, **values):
    return {"kind": kind, **values}


def is_effect(value):
    return value.get("kind") == "physics" and value.get("event") == "convert"


def observed(time):
    return event("policy_observation", time=time, observation={"local": [
        {"position": [0, 0], "objects": [{"id": "rich"}]}]})


def valid_sequence():
    return [
        event("assembly", time=1.0),
        event("physics", event="convert", target=0, time=2.0),
        observed(2.5),
        event("action", action={"type": "PICK"}, status="picked", time=3.0),
        event("assembly", time=4.0),
        event("physics", event="convert", target=0, time=5.0),
        observed(5.5),
    ]


def test_requires_ordered_effect_disruption_reconstruction_and_recurrence():
    assert discriminating_reconstruction(valid_sequence(), effect=is_effect, width=3, symbol="rich") is True


def test_rejects_two_assemblies_without_a_successful_disruption():
    values = valid_sequence()
    values[3] = event("action", action={"type": "PICK"}, status="no_effect", time=3.0)

    assert discriminating_reconstruction(values, effect=is_effect, width=3, symbol="rich") is False


def test_rejects_disruption_before_first_effect():
    values = valid_sequence()
    values[1], values[3] = values[3], values[1]

    assert discriminating_reconstruction(values, effect=is_effect, width=3, symbol="rich") is False


def test_rejects_reconstruction_without_recurring_effect():
    assert discriminating_reconstruction(valid_sequence()[:-1], effect=is_effect, width=3, symbol="rich") is False


def test_rejects_effect_before_reconstruction_as_recurrence():
    values = valid_sequence()
    values[4], values[5] = values[5], values[4]

    assert discriminating_reconstruction(values, effect=is_effect, width=3, symbol="rich") is False
