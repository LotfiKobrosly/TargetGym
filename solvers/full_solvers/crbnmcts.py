"""
This file implements a Continuous Reward-based Nested Monte Carlo Tree Search
"""

from copy import deepcopy
import numpy as np

from classes.environment import EnvironmentWrapper
from utils.models import code


def crbnmcts(
    environment: EnvironmentWrapper,
    level: int = 1,
    bandwidth: int = 20,
    action: np.ndarray = None,
):
    if level == 0:
        environment.step(action)
        return (
            environment.sequence,
            environment.actions,
            environment.score,
        )

    else:
        if not (action is None):
            environment.step(action)
        current_timestamp = environment.current_timestamp
        best_sequence = environment.best_sequence[:]
        best_actions = environment.best_actions[:]
        best_overall_score = environment.best_score
        while not environment.is_final():
            actions_list = environment.sample_random_action(bandwidth)
            best_action = None
            best_score = environment.best_score
            current_actions = environment.actions[:]
            for new_action in actions_list:
                sequence, actions, score = crbnmcts(
                    environment,
                    level=level - 1,
                    bandwidth=bandwidth,
                    action=new_action,
                )
                if score > best_score:
                    best_action = new_action
                    best_score = score

                if environment.is_final() and (score > best_overall_score):
                    best_overall_score = score
                    best_sequence = sequence[:]
                    best_actions = actions[:]

                # Resetting
                environment.reset()
                for timestamp in range(current_timestamp):
                    environment.step(current_actions[timestamp])
                environment.best_score = best_overall_score
                environment.best_sequence = best_sequence[:]
                environment.best_actions = best_actions[:]
            if best_action is None:
                try:
                    best_action = environment.best_actions[current_timestamp]
                except IndexError:
                    print(score)
                    print(best_overall_score)
                    print(environment.best_score)
                    print(current_timestamp)
                    print(len(environment.best_actions))
                    raise IndexError("Out of range")
            environment.step(best_action)
            current_timestamp += 1

        return (
            best_sequence,
            best_actions,
            best_overall_score,
        )
