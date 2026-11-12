from copy import deepcopy
import numpy as np
import jax
import target_gym

from classes.environment import EnvironmentWrapper
from classes.continuous_dictionary import ContinuousGaussianDictionary
from classes.gaussian_kernel import GaussianKernel
from utils.models import code
from utils.constants import (
    ACTION_KEY,
    RESET_KEY,
    RELEVANCE_THRESHOLD,
    HALF_LIFE_DIVIDER,
)


class GcNRPA_Policy:

    def __init__(
        self,
        environment: EnvironmentWrapper,
        level: int,
        n_policies: int,
        residual_range: float = 0.05,
        half_life_divider: int = 10,
    ):
        self.environment = environment
        self.level = level
        self.n_policies = n_policies
        self.residual_range = residual_range
        self.half_life_divider = half_life_divider
        self.residual_policy = None
        self.current_timestamp = 0
        self.pid = self.environment.get_pid()

    def reset(self, jax_keys: dict = None):

        if not (jax_keys is None):
            self.environment.jax_keys = jax_keys

        self.pid.reset()
        self.environment.reset()
        self.current_timestamp = 0

    def train_full_gcnrpa_policy(self):
        dimension = len(self.environment.current_state)
        if self.residual_policy is None:

            kernel_radius = 0.1
            self.residual_policy = instantiate_policy(
                kernel_radius=kernel_radius,
            )

        self.pid = self.environment.get_pid()
        best_score, best_sequence, best_actions = gcnrpa(
            environment=self.environment,
            pid=self.pid,
            level=self.level,
            n_policies=self.n_policies,
            policy=self.residual_policy,
            residual_range=self.residual_range,
            current_iteration=0,
            half_life_divider=self.half_life_divider,
        )

    def __call__(self, observation):
        assert not (self.residual_policy is None), "GcNRPA has not been trained yet"
        controller_action = self.pid(np.asarray(observation)).flatten()
        residual_action = self.residual_policy[
            (*observation, self.current_timestamp)
        ].flatten()
        self.current_timestamp += 1
        return controller_action + self.residual_range * residual_action


def instantiate_policy(
    kernel_radius: float = None,
):
    assert not (
        kernel_radius is None
    ), "Value of kernel_radius must be specified when Gaussian policy is chosen"
    return ContinuousGaussianDictionary(kernel_radius)


def adapt_policy(
    policy: dict = None,
    actions: list = None,
    sequence: list = None,
    learning_rate: float = None,
):
    if policy:
        # Get dimension of the keys
        dimension = len(sequence[0])

        # Get all values
        kernel_radius = np.exp(1 / len(policy)) / 10

        # Adapting visited states
        for timestamp, state in enumerate(sequence[:-1]):
            new_move = actions[timestamp]
            current_state = code((*state, timestamp))
            if current_state in policy.keys():
                new_move = np.reshape(new_move, policy[current_state].shape)
                policy[current_state] += learning_rate * (
                    new_move - policy[current_state]
                )
            else:
                policy[current_state] = new_move
            assert not np.any(
                np.isnan(policy[current_state])
            ), "NaN in adaptation " + str(policy[current_state])

        # Adapting nearby states
        for state in policy.keys():
            weights = np.zeros(len(actions))
            gaussian_filter = GaussianKernel(state, kernel_radius)
            # Getting relevant points
            relevant_actions, weights = list(), list()
            for timestamp, point in enumerate(sequence[:-1]):
                full_state = *point, timestamp
                distance = np.linalg.norm(np.array(state) - np.array(full_state))
                if (distance <= RELEVANCE_THRESHOLD) and (distance > 1e-3):
                    relevant_actions.append(actions[timestamp])
                    weight_to_add = gaussian_filter.pdf(full_state)
                    if weight_to_add == 0:
                        print("Point:", full_state, ", vs state:", state)
                        raise ValueError("Issue in computing pdf value of nearby state")
                    weights.append(weight_to_add)

            if relevant_actions:
                # assert not (np.any(np.isnan(relevant_actions))), "NaN in relevant_actions " + str(relevant_actions)
                weights = np.array(weights)
                weights /= np.sum(weights)
                assert not (
                    np.any(np.isnan(weights))
                ), "NaN in weights in adaptation " + str(weights)
                new_move = weights.T @ np.array(relevant_actions)
                assert not np.any(np.isnan(new_move)), "NaN in computing new move"
                new_move = np.reshape(new_move, policy[state].shape)
                policy[state] += learning_rate * (new_move - policy[state])
    else:
        for timestamp, state in enumerate(sequence[:-1]):
            policy[code((*state, timestamp))] = actions[timestamp]


def gcnrpa(
    environment: EnvironmentWrapper,
    pid: target_gym.experts.pid.StatefulCascadedAltitudePID = None,
    level: int = 1,
    n_policies: int = 100,
    policy: dict = None,
    residual_range: float = 0.05,
    current_iteration: int = 0,
    half_life_divider: int = None,
    *args,
    **kwargs,
):
    if level == 0:
        actions_sequence = list()
        sampling_radius = np.exp(-current_iteration / (n_policies / half_life_divider))
        environment.reset()
        while not environment.is_final():
            pid_action = pid(np.asarray(environment.current_state))
            if len(policy) > 0:
                center = policy[
                    code((*environment.current_state, environment.current_timestamp))
                ]
                assert not np.any(np.isnan(center)), "NaN found in policy"
                action = jax.random.multivariate_normal(
                    key=ACTION_KEY,
                    mean=center,
                    cov=sampling_radius * np.eye(max(center.shape)),
                )
            else:
                action = jax.random.uniform(
                    key=ACTION_KEY, shape=pid_action.shape, minval=-1.0, maxval=1.0
                )

            environment.step(pid_action + residual_range * action)
            actions_sequence.append(action)
        return environment.score, environment.sequence, actions_sequence

    else:
        new_policy = deepcopy(policy)
        best_score = environment.best_score
        best_sequence = environment.best_sequence[:]
        best_actions = environment.best_actions[:]
        learning_rate = np.sqrt(1 / environment.horizon)
        for iteration in range(n_policies):
            environment.reset()
            pid.reset()
            score, sequence, actions = gcnrpa(
                environment,
                pid,
                level - 1,
                n_policies,
                new_policy,
                residual_range,
                iteration,
                half_life_divider,
            )
            if score > best_score:
                best_score = score
                best_sequence = sequence[:]
                best_actions = actions[:]
            adapt_policy(
                new_policy,
                best_actions,
                best_sequence,
                learning_rate,
            )
        adapt_policy(
            policy,
            best_actions,
            best_sequence,
            learning_rate,
        )

        return best_score, best_sequence, best_actions


def run_residual_gcnrpa(
    environment: EnvironmentWrapper,
    level: int = 1,
    residual_range: float = 0.05,
    n_policies: int = 100,
    half_life_divider: int = 10,
):
    dimension = len(environment.current_state)
    kernel_radius = 0.1
    policy = instantiate_policy(
        kernel_radius=kernel_radius,
    )
    pid = environment.get_pid()
    # environment.allow_truncation()
    best_score, best_sequence, best_actions = gcnrpa(
        environment,
        pid,
        level,
        n_policies,
        policy,
        residual_range,
        0,
        half_life_divider,
    )
    # environment.disallow_truncation()
    return best_sequence, best_actions, best_score
