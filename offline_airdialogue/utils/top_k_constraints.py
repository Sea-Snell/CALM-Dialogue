from typing import Any, Dict, List, Generator
import numpy as np
import heapq

Probability = List[float]

class HoleState:
    def __init__(self, probs: Probability, curr_state: int=None):
        self.probs = probs
        self.priority = np.argsort(self.probs)[::-1]
        self.curr_state = 0 if curr_state is None else curr_state
        self.index = self.priority[self.curr_state]
        self.neg_log_prob = -np.log(self.probs[self.index])

    def next_state(self):
        if self.curr_state + 1 >= len(self.priority):
            return None
        return HoleState(self.probs, curr_state=self.curr_state + 1)

class ScaffoldState:
    def __init__(self, id: int, ref: any, hole_states: List[HoleState], scaffold_neg_log_prob: float):
        self.ref = ref
        self.id = id
        self.scaffold_neg_log_prob = scaffold_neg_log_prob
        self.hole_states = tuple(hole_states)
        self.indices = list(map(lambda x: int(x.index), self.hole_states))
        self.total_neg_log_prob = sum(map(lambda x: x.neg_log_prob,
                                          self.hole_states)) + self.scaffold_neg_log_prob

    def next_state(self) -> List['ScaffoldState']:
        next_states = []
        for i in range(len(self.hole_states)):
            next_state = self.hole_states[i].next_state()
            if next_state is None:
                continue
            next_states.append(ScaffoldState(self.id, self.ref,
                                             self.hole_states[:i] + tuple([next_state]) + self.hole_states[i+1:],
                                             self.scaffold_neg_log_prob))
        return next_states

    def __eq__(self, other):
        return self.id == other.id and all(map(lambda x: x[0].curr_state == x[1].curr_state, zip(self.hole_states, other.hole_states)))

    def __hash__(self):
        return hash(tuple([self.id] + list(map(lambda x: x.curr_state, self.hole_states))))

def best_first_enumerate_states(start_nodes: List[ScaffoldState]) -> Generator[ScaffoldState, None, None]:
    pq = []
    visited = set()
    curr_id = 0
    for start in start_nodes:
        heapq.heappush(pq, (start.total_neg_log_prob, curr_id, start))
        curr_id += 1
    while len(pq) > 0:
        score, _, state = heapq.heappop(pq)
        if state in visited:
            continue
        visited.add(state)
        yield state
        for next_state in state.next_state():
            heapq.heappush(pq, (next_state.total_neg_log_prob, curr_id, next_state))
            curr_id += 1

def top_k_constraints(constraint_probs: Dict[str, Probability], constraint_idx2val: Dict[str, List[Any]], k_limit: int = None):
    keys = list(constraint_probs.keys())
    state = ScaffoldState(0, None, [HoleState(constraint_probs[k]) for k in keys], 0.0)
    for i, items in enumerate(best_first_enumerate_states([state])):
        if k_limit is not None and i >= k_limit:
            break
        new_constraints = {k: constraint_idx2val[k][idx] for k, idx in zip(keys, items.indices)}
        prob = np.exp(-items.total_neg_log_prob)
        yield new_constraints, prob