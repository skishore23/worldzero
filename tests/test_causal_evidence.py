from __future__ import annotations

from worldzero.causal_evidence import discriminating_reconstruction


def event(kind, **values):
    return {"kind": kind, **values}


def is_effect(value):
    return value.get("kind") == "physics" and value.get("event") == "convert"


def valid_sequence():
    return [
        event("assembly", time=1.0),
        event("physics", event="convert", time=2.0),
        event("action", action={"type": "PICK"}, status="picked", time=3.0),
        event("assembly", time=4.0),
        event("physics", event="convert", time=5.0),
    ]


def test_requires_ordered_effect_disruption_reconstruction_and_recurrence():
    assert discriminating_reconstruction(valid_sequence(), effect=is_effect) is True


def test_rejects_two_assemblies_without_a_successful_disruption():
    values = valid_sequence()
    values[2] = event("action", action={"type": "PICK"}, status="no_effect", time=3.0)

    assert discriminating_reconstruction(values, effect=is_effect) is False


def test_rejects_disruption_before_first_effect():
    values = [valid_sequence()[0], valid_sequence()[2], valid_sequence()[1],
              valid_sequence()[3], valid_sequence()[4]]

    assert discriminating_reconstruction(values, effect=is_effect) is False


def test_rejects_reconstruction_without_recurring_effect():
    assert discriminating_reconstruction(valid_sequence()[:-1], effect=is_effect) is False


def test_rejects_effect_before_reconstruction_as_recurrence():
    values = valid_sequence()
    values[3], values[4] = values[4], values[3]

    assert discriminating_reconstruction(values, effect=is_effect) is False
