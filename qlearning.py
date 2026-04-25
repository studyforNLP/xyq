import random
from typing import Dict, List, Tuple

import numpy as np

from state_space import ACTIONS


State = Tuple[int, ...]
Action = Tuple[str, int, int]


def init_q_table(states, actions) -> Dict:
    """Initialize Q as {state: {action_tuple: q_value}}."""
    return {state: {} for state in states}


def epsilon_greedy_action(
    Q: Dict,
    state: State,
    actions_list: List[Action],
    epsilon: float = 0.1,
):
    """Select an action with epsilon-greedy policy."""
    if not actions_list:
        return None
    if np.random.rand() < epsilon:
        return random.choice(actions_list)

    best_action = None
    best_q = float("-inf")
    for action in actions_list:
        q_value = Q.get(state, {}).get(action, 0.0)
        if q_value > best_q:
            best_q = q_value
            best_action = action
    return best_action if best_action is not None else random.choice(actions_list)


def update_q(
    Q: Dict,
    state: State,
    action: Action,
    reward: float,
    next_state: State,
    alpha: float = 0.1,
    gamma: float = 0.9,
) -> None:
    """One-step Q-learning update."""
    action_type = action[0] if isinstance(action, tuple) else action
    if action_type not in ACTIONS:
        return

    if state not in Q:
        Q[state] = {}
    if next_state not in Q:
        Q[next_state] = {}

    old_value = Q[state].get(action, 0.0)
    next_max = max(Q[next_state].values()) if Q[next_state] else 0.0
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    Q[state][action] = new_value
