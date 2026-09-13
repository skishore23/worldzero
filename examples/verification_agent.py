"""Optional observation-only controls for the verification calibration study.

These hand-authored policies have an adjacent-pair and resource-transformation
prior. They are not general discovery agents and cannot identify preservation
from persistence alone. No world seed, family identity, or evaluator event is
read by these agents. The retain ablation uses the exact same search policy.
"""

from worldzero.policies import (
    BlindManipulatorPolicy, ExperimenterPolicy, ForagerPolicy, action, distance,
    toward,
)
from worldzero.util import derive_seed


class VerificationPolicy(ExperimenterPolicy):
    """Test a provisional transformation by removing and replacing a component.

    Recurrence requires an observed consumable to change identity at the same
    visible position after reconstruction. An old transformed resource left on
    the ground does not qualify. This is still a behavioral positive control:
    it does not estimate an off/on treatment effect or establish understanding.
    """

    name = "verification"

    def __init__(self, seed=0):
        super().__init__(seed)
        self.verification_phase = "search"
        self.rebuild_position = None
        self.off_since = None
        self.previous_food = {}
        self.output_ids = set()
        self.verified = False

    def decide(self, observation):
        if self.verification_phase == "search":
            response = super().decide(observation)
            if self.confirmed:
                self.verification_phase = "approach"
                self.output_ids = {
                    token for position, token in self.food
                    if token not in self.baseline_ids
                    and distance(position, self.working_site) <= 3
                }
            return response

        self.update(observation)
        position = tuple(observation["position"])
        now_food = dict(self.food)
        if self.verification_phase == "recurrence":
            recurrence = any(
                token in self.output_ids
                and p in self.previous_food
                and self.previous_food[p] not in self.output_ids
                and distance(p, self.working_site) <= 3
                for p, token in now_food.items()
            )
            if recurrence:
                self.verified = True
                self.verification_phase = "use"
        self.previous_food = now_food

        # Identical emergency threshold to the search/retain policy.
        if any(p == position for p, _ in self.food):
            return action("CONSUME", "Maintain energy while testing recurrence.")
        if observation["energy"] < 6:
            return self.forage(observation)

        if self.verification_phase == "approach":
            target = self.modules.get(self.carry)
            if target is None:
                return self.forage(observation)
            if position != target:
                return action("MOVE", direction=toward(position, target))
            self.rebuild_position = position
            self.verification_phase = "remove"
            return action("PICK", "Disrupt the candidate mechanism.")

        if self.verification_phase == "remove":
            if observation["inventory"] != self.carry:
                self.verification_phase = "approach"
                return action("WAIT", duration=0.5)
            self.off_since = observation["time"]
            self.verification_phase = "off"

        if self.verification_phase == "off":
            if observation["time"] - self.off_since < 6:
                return action("WAIT", "Keep one component removed.", duration=2.0)
            self.verification_phase = "rebuild"

        if self.verification_phase == "rebuild":
            if observation["inventory"] != self.carry:
                # Only enter recurrence after a successful replacement.
                if self.modules.get(self.carry) == self.rebuild_position:
                    self.verification_phase = "recurrence"
                    self.previous_food = now_food
                else:
                    return self.forage(observation)
            elif position != self.rebuild_position:
                return action("MOVE", direction=toward(position, self.rebuild_position))
            elif observation["legal_actions"]["DROP"]["available"]:
                return action("DROP", "Reconstruct the candidate mechanism.")

        if self.verification_phase in {"recurrence", "use"}:
            if distance(position, self.working_site) > 2:
                return action("MOVE", direction=toward(position, self.working_site))
            if observation["energy"] > 12:
                return action("WAIT", "Observe the reconstructed arrangement.", duration=2.0)
            return self.forage(observation, stay=self.working_site)
        return self.forage(observation)


class StudyAgent:
    def __init__(self, kind):
        self.kind = kind

    def reset(self, context):
        classes = {
            "verify": VerificationPolicy,
            "retain": ExperimenterPolicy,
            "blind": BlindManipulatorPolicy,
            "forager": ForagerPolicy,
        }
        # Matches make_policy's seed derivation for the existing baselines.
        self.policy = classes[self.kind](derive_seed(context["agent_seed"], "policy-v2"))

    def act(self, observation):
        response = self.policy.decide(observation)
        supported = (
            self.policy.verified if self.kind == "verify"
            else getattr(self.policy, "confirmed", False)
        )
        return {
            "action": response["action"],
            "finding": {"status": "supported" if supported else "insufficient_evidence"},
        }

    def observe_result(self, result):
        pass

    def close(self):
        pass


def create_verifier():
    return StudyAgent("verify")


def create_retainer():
    return StudyAgent("retain")


def create_blind():
    return StudyAgent("blind")


def create_forager():
    return StudyAgent("forager")
