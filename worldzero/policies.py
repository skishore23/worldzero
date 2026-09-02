"""Reference policies. All decide() methods receive ONLY JSON observations.

The experimenter is an explicitly hand-designed intervention-search control,
not an LLM and not a claim that experimentation emerged. The informed control
gets the active pair once at birth and is labeled privileged, not optimal.
"""
from __future__ import annotations
from itertools import combinations
from typing import Any, Protocol
import random


class Policy(Protocol):
    name: str
    def decide(self, observation: dict[str,Any]) -> dict[str,Any]: ...


def action(kind: str, memory: str = "", **kwargs: Any) -> dict[str,Any]:
    return {"action":{"type":kind,**kwargs},"memory":memory}


def distance(a: tuple[int,int], b: tuple[int,int]) -> int:
    return abs(a[0]-b[0])+abs(a[1]-b[1])


def toward(a: tuple[int,int], b: tuple[int,int]) -> str:
    if b[0] < a[0]: return "N"
    if b[0] > a[0]: return "S"
    return "E" if b[1] > a[1] else "W"


class RandomPolicy:
    name = "random"
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
    def decide(self, observation: dict[str,Any]) -> dict[str,Any]:
        options = [action("MOVE",direction=d) for d in ("N","E","S","W")]
        options += [action("PICK"),action("DROP"),action("CONSUME"),action("WAIT",duration=2)]
        return self.rng.choice(options)


class ForagerPolicy:
    """Observation-only, non-constructing baseline; learns consumable values."""
    name = "forager"
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.values: dict[str,float] = {}
        self.visits: dict[tuple[int,int],int] = {}
        self.terrain: dict[tuple[int,int],int] = {}
        self.modules: dict[str,tuple[int,int]] = {}
        self.resource_ids: set[str] = set()
        self.food: list[tuple[tuple[int,int],str]] = []
        self.last_seen_position: tuple[int,int] | None = None

    def update(self, o: dict[str,Any]) -> None:
        pos = tuple(o["position"])
        self.visits[pos] = self.visits.get(pos,0)+1
        result = o.get("last_result",{})
        if result.get("status") == "consumed" and result.get("object_id"):
            self.values[result["object_id"]] = float(result["gross_energy"])
        self.food = []
        visible_positions = {tuple(c["position"]) for c in o["local"]}
        self.modules = {k:p for k,p in self.modules.items() if p not in visible_positions}
        for cell in o["local"]:
            p = tuple(cell["position"])
            self.terrain[p] = cell["surface"]
            for obj in cell["objects"]:
                if obj["consume"]:
                    self.food.append((p,obj["id"])); self.resource_ids.add(obj["id"])
                if obj["pick"]:
                    self.modules[obj["id"]] = p
        if o["inventory"]:
            self.modules.pop(o["inventory"],None)

    def forage(self, o: dict[str,Any], *, stay: tuple[int,int] | None = None) -> dict[str,Any]:
        pos = tuple(o["position"])
        here = [token for p,token in self.food if p==pos]
        if here:
            return action("CONSUME", "Consume an available item; update its observed return.")
        if self.food:
            def score(item: tuple[tuple[int,int],str]) -> float:
                p,token = item
                value = self.values.get(token,2.0)
                return value/(1+distance(pos,p))
            p,token = max(self.food,key=score)
            return action("MOVE", "Approach an observed consumable.", direction=toward(pos,p))
        if stay is not None:
            if distance(pos,stay)>2:
                return action("MOVE","Return near a previously useful location.",direction=toward(pos,stay))
            return action("WAIT","Allow environmental processes to continue.",duration=4.0)
        height,width = o["bounds"]
        options = [("N",(pos[0]-1,pos[1])),("E",(pos[0],pos[1]+1)),
                   ("S",(pos[0]+1,pos[1])),("W",(pos[0],pos[1]-1))]
        options = [(d,p) for d,p in options if 0<=p[0]<height and 0<=p[1]<width]
        d,_ = max(options,key=lambda dp:(2*self.terrain.get(dp[1],1)-0.4*self.visits.get(dp[1],0),self.rng.random()))
        return action("MOVE","Search another visible location.",direction=d)

    def decide(self, observation: dict[str,Any]) -> dict[str,Any]:
        self.update(observation)
        return self.forage(observation)


class ExperimenterPolicy(ForagerPolicy):
    """Generic pair-intervention prior; the useful pair is NOT supplied.

    Confirmation is deliberately modest: a new consumable identity appears
    near the intervention, rather than an evaluator's hidden success bit.
    This is a calibration policy, NOT evidence of unprompted scientific agency.
    """
    name = "experimenter"
    def __init__(self, seed: int = 0, *, informed_pair: tuple[str,str] | None = None) -> None:
        super().__init__(seed)
        self.informed_pair = informed_pair
        if informed_pair is not None:
            self.name = "informed"
        self.tested: set[tuple[str,str]] = set()
        self.current: tuple[str,str] | None = None
        self.carry: str | None = None
        self.anchor: str | None = None
        self.target: tuple[int,int] | None = None
        self.phase = "select"
        self.observe_start = 0.0
        self.baseline_ids: set[str] = set()
        self.confirmed = False
        self.confirmed_pair: tuple[str,str] | None = None
        self.working_site: tuple[int,int] | None = None
        self.trials = 0

    def _choose(self, o: dict[str,Any]) -> bool:
        pos = tuple(o["position"])
        labels = sorted(self.modules)
        pairs = [tuple(sorted(self.informed_pair))] if self.informed_pair else list(combinations(labels,2))
        choices = []
        for pair in pairs:
            if pair in self.tested or any(x not in self.modules for x in pair):
                continue
            for carry,anchor in (pair,pair[::-1]):
                cp,ap = self.modules[carry],self.modules[anchor]
                h,w = o["bounds"]
                candidates = [(ap[0]-1,ap[1]),(ap[0]+1,ap[1]),(ap[0],ap[1]-1),(ap[0],ap[1]+1)]
                occupied = set(self.modules.values())
                for p in candidates:
                    if not (0<=p[0]<h and 0<=p[1]<w) or p in occupied:
                        continue
                    neighbors = [(p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1)]
                    productivity = sum(self.terrain.get(q,0) for q in neighbors)
                    cost = distance(pos,cp)+distance(cp,p)
                    choices.append((2.5*productivity-cost*0.25,pair,carry,anchor,p))
        if not choices:
            return False
        _,self.current,self.carry,self.anchor,self.target = max(choices,key=lambda row:row[0])
        self.phase = "build"
        self.trials += 1
        return True

    def decide(self, o: dict[str,Any]) -> dict[str,Any]:
        previous_ids = set(self.resource_ids)
        self.update(o)
        pos = tuple(o["position"])
        if self.confirmed:
            # A confirmation is not omniscient knowledge: component decay may
            # invalidate it. Observe losses and permit new experiments.
            if self.confirmed_pair and all(x in self.modules for x in self.confirmed_pair):
                p,q = (self.modules[x] for x in self.confirmed_pair)
                if distance(p,q)!=1:
                    self.confirmed = False
                    self.tested.discard(tuple(sorted(self.confirmed_pair)))
                    self.phase = "select"
            if self.confirmed:
                if o["energy"] > 12 and not any(p==pos for p,_ in self.food):
                    return action("WAIT","Retain the arrangement; test continued output by waiting.",duration=4.0)
                return self.forage(o,stay=self.working_site)
        # Eat when already on food; emergency foraging is not a privileged rescue.
        if any(p==pos for p,_ in self.food):
            return action("CONSUME","Maintain energy during the investigation.")
        if o["energy"] < 6:
            return self.forage(o)
        if self.phase == "observe":
            novel = [(p,tok) for p,tok in self.food if tok not in self.baseline_ids and
                     self.working_site is not None and distance(p,self.working_site)<=3]
            if novel or self.informed_pair is not None:
                self.confirmed = True; self.confirmed_pair = self.current
                out = action("WAIT","Observed a changed consumable outcome near the intervention; retain provisionally.",duration=2.0)
                out["belief"] = {"pair":list(self.current),"evidence":"novel local consumable" if novel else "informed control", "provisional":True}
                return out
            if o["time"]-self.observe_start < 18:
                return action("WAIT","Observe the candidate arrangement before changing it again.",duration=3.0)
            self.tested.add(self.current)
            self.phase = "select"; self.current = None
        if self.phase == "select":
            if o["inventory"] is not None:
                occupied = set(self.modules.values())
                if pos not in occupied:
                    return action("DROP","Put down the previous experimental component.")
                return self.forage(o)
            if not self._choose(o):
                return self.forage(o)
        if self.phase == "build":
            if self.anchor not in self.modules:
                self.phase = "select"
                return self.forage(o)
            anchor_pos = self.modules[self.anchor]
            if self.carry in self.modules and distance(self.modules[self.carry],anchor_pos)==1:
                self.phase = "observe"; self.observe_start = o["time"]
                self.baseline_ids = previous_ids or set(self.resource_ids)
                self.working_site = anchor_pos
                return action("WAIT","Candidate placed; establish an observation window.",duration=2.0)
            if o["inventory"] == self.carry:
                if pos == self.target:
                    return action("DROP","Place the component in the candidate configuration.")
                return action("MOVE","Carry the component to the experimental site.",direction=toward(pos,self.target))
            if o["inventory"] is not None:
                if pos not in self.modules.values():
                    return action("DROP","Clear inventory before continuing.")
                return self.forage(o)
            if self.carry not in self.modules:
                self.phase = "select"
                return self.forage(o)
            cp = self.modules[self.carry]
            if pos == cp:
                return action("PICK","Move one component as an intervention.")
            return action("MOVE","Approach the component selected for intervention.",direction=toward(pos,cp))
        return self.forage(o)


class BlindManipulatorPolicy(ForagerPolicy):
    """A deterministic, observation-only negative control for rearrangement.

    The schedule and RNG are intentionally fixed before outcomes are observed.
    This policy has no access to the active pair, evaluator state, or world.
    """
    name = "blind-manipulator"
    schedule = (24, 64, 104)

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.decision_count = 0
        self.cycle_index = 0
        self.phase = "idle"
        self.target_id: str | None = None
        self.target_position: tuple[int, int] | None = None
        self.walk_remaining = 0
        self.manipulation_cycles_started = 0
        self.manipulation_cycles_completed = 0

    @staticmethod
    def _visible_portable(o: dict[str, Any]) -> dict[str, tuple[int, int]]:
        portable = {}
        for cell in o["local"]:
            position = tuple(cell["position"])
            for item in cell["objects"]:
                if item.get("pick"):
                    portable[item["id"]] = position
        return portable

    def _clear_target(self) -> None:
        self.phase = "idle"
        self.target_id = None
        self.target_position = None
        self.walk_remaining = 0

    def _reconcile_pending_action(self, o: dict[str, Any]) -> None:
        result = o.get("last_result", {})
        recorded_action = result.get("action", {})
        if self.phase == "await_pick":
            if (recorded_action.get("type") == "PICK" and result.get("status") == "picked"
                    and result.get("object_id") == self.target_id
                    and o["inventory"] == self.target_id):
                self.phase = "carry"
                self.walk_remaining = self.rng.randint(2, 8)
                self.cycle_index += 1
                self.manipulation_cycles_started += 1
            else:
                self._clear_target()
        elif self.phase == "await_drop":
            if (recorded_action.get("type") == "DROP" and result.get("status") == "dropped"
                    and result.get("object_id") == self.target_id and o["inventory"] is None):
                self.manipulation_cycles_completed += 1
                self._clear_target()
            elif o["inventory"] is not None:
                self.phase = "carry"
            else:
                self._clear_target()

    def after_step(self, observation: dict[str, Any] | None, result: dict[str, Any]) -> None:
        """Reconcile a public action result, including at an episode boundary."""
        if observation is not None and observation.get("last_result") == result:
            self._reconcile_pending_action(observation)

    def _select_target(self, o: dict[str, Any]) -> bool:
        if self.cycle_index >= len(self.schedule):
            return False
        if self.decision_count < self.schedule[self.cycle_index] or o["energy"] < 8:
            return False
        portable = self._visible_portable(o)
        if not portable:
            return False
        position = tuple(o["position"])
        nearest = min(distance(position, target) for target in portable.values())
        choices = sorted(item_id for item_id, target in portable.items()
                         if distance(position, target) == nearest)
        self.target_id = self.rng.choice(choices)
        self.target_position = portable[self.target_id]
        self.phase = "approach"
        return True

    def _approach_target(self, o: dict[str, Any]) -> dict[str, Any] | None:
        if self.phase != "approach" or self.target_id is None:
            return None
        portable = self._visible_portable(o)
        if self.target_id not in portable:
            self._clear_target()
            return None
        self.target_position = portable[self.target_id]
        position = tuple(o["position"])
        if position == self.target_position:
            if o["legal_actions"]["PICK"]["available"]:
                self.phase = "await_pick"
                return action("PICK", "Blind scheduled manipulation pick.")
            return self.forage(o)
        if o["inventory"] is not None:
            self._clear_target()
            return self.forage(o)
        return action("MOVE", "Approach a currently visible portable object.",
                      direction=toward(position, self.target_position))

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.decision_count += 1
        self.update(observation)
        self._reconcile_pending_action(observation)

        if self.phase == "carry" and observation["inventory"] is not None:
            directions = observation["legal_actions"]["MOVE"]["directions"]
            if self.walk_remaining > 0 and directions:
                self.walk_remaining -= 1
                return action("MOVE", "Blind scheduled carry walk.",
                              direction=self.rng.choice(directions))
            if observation["legal_actions"]["DROP"]["available"]:
                self.phase = "await_drop"
                return action("DROP", "Blind scheduled manipulation drop.")
        elif self.phase == "carry":
            self._clear_target()

        targeted = self._approach_target(observation)
        if targeted is not None:
            return targeted
        if self._select_target(observation):
            targeted = self._approach_target(observation)
            if targeted is not None:
                return targeted
        return self.forage(observation)


class ReplayPolicy:
    name = "replay"
    def __init__(self, decisions: list[dict[str,Any]]) -> None:
        self.decisions = iter(decisions)
    def decide(self, observation: dict[str,Any]) -> dict[str,Any]:
        return next(self.decisions)
