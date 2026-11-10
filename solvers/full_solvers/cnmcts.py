from copy import deepcopy
import numpy as np

from classes.environment import EnvironmentWrapper
from utils.models import code


def cnmcts(environment: EnvironmentWrapper, level: int = 1, bandwidth: int = 20):
    if level == 0:
        while not environment.is_final():
            action = environment.sample_random_action(n_samples=1)
            environment.step(action)
        return environment.sequence, environment.actions, environment.score

    else:
        current_timestamp = environment.current_timestamp
        best_sequence = environment.best_sequence[:]
        best_actions = environment.best_actions[:]
        best_score = environment.best_score
        while not environment.is_final():
            actions_list = environment.sample_random_action(bandwidth)
            current_actions = environment.actions[:]
            for action in actions_list:
                environment.step(action)
                sequence, actions, score = cnmcts(environment, level - 1, bandwidth)
                if score > best_score:
                    best_sequence = sequence[:]
                    best_actions = actions[:]
                    best_score = score

                # Resetting
                environment.reset()
                for timestamp in range(current_timestamp):
                    environment.step(current_actions[timestamp])
                environment.best_score = best_score
                environment.best_sequence = best_sequence[:]
                environment.best_actions = best_actions[:]

            # Do best action
            environment.step(best_actions[current_timestamp])
            current_timestamp += 1
        return (
            best_sequence,
            best_actions,
            best_score,
        )
