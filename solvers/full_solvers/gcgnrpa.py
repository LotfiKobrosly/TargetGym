from copy import deepcopy
import numpy as np
import jax

from classes.environment import EnvironmentWrapper
from classes.continuous_dictionary import *
from utils.models import code
from utils.constants import ACTION_KEY, RELEVANCE_THRESHOLD, HALF_LIFE_DIVIDER
from solvers.cnrpa import adapt_policy, instantiate_policy


def gcgnrpa(
    environment: EnvironmentWrapper,
    level: int = 1,
    n_policies: int = 100,
    policy: dict = None,
    n_visits: dict = None,
    current_iteration: int = 0,
    full_area: float = None,
    half_life_divider: int = None,
    tau: float = 1.0,
    bias_provider=None,
):
    if level == 0:
        sampling_radius = np.exp(-current_iteration / (n_policies / half_life_divider))
        if isinstance(policy, ContinuousByRegionDictionary):
            environment.allow_truncation()
        environment.reset()
        while not environment.is_final():

            # Bias
            bias = bias_provider(np.asarray(environment.current_state))

            # Policy or random
            if len(policy) > 0:
                policy_action = policy[code(environment.current_state)]
                assert not np.any(np.isnan(policy_action)), "NaN found in policy"

            else:
                policy_action = environment.sample_random_action(n_samples=1)

            # Combination
            if jax.random.uniform(ACTION_KEY) < (1 / tau):
                center = bias
            else:
                center = policy_action
            action = jax.random.multivariate_normal(
                key=ACTION_KEY,
                mean=center,
                cov=sampling_radius * np.eye(max(center.shape)),
            )
            environment.step(action)
        return environment.score, environment.sequence, environment.actions

    else:
        new_policy = deepcopy(policy)
        new_n_visits = deepcopy(n_visits)
        best_score = environment.best_score
        best_sequence = environment.best_sequence[:]
        best_actions = environment.best_actions[:]
        learning_rate = np.sqrt(1 / environment.horizon)
        for iteration in range(n_policies):
            environment.reset()
            score, sequence, actions = gcgnrpa(
                environment,
                level - 1,
                n_policies,
                new_policy,
                new_n_visits,
                iteration,
                full_area,
                half_life_divider,
                tau,
                bias_provider,
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
                new_n_visits,
                full_area,
            )
        adapt_policy(
            policy, best_actions, best_sequence, learning_rate, n_visits, full_area
        )

        return best_score, best_sequence, best_actions


def run_gcgnrpa(
    environment: EnvironmentWrapper,
    level: int = 1,
    n_policies: int = 100,
    policy_type: str = "by_region",
    half_life_divider: int = 10,
    tau: float = 1.0,
):
    dimension = len(environment.current_state)
    kernel_radius = 0.3
    policy = instantiate_policy(
        policy_type,
        kernel_radius=kernel_radius,
    )
    pid = environment.environment.make_pid()
    if policy_type == "by_region":

        start_region = (
            tuple([-2 for _ in range(dimension)]),
            tuple([2 for _ in range(dimension)]),
        )
        n_visits = dict()
        policy[start_region] = environment.sample_random_action(n_samples=1)
        n_visits[start_region] = {"n_visits": 0, "threshold": 1}
        full_area = get_region_area(start_region)
        environment.allow_truncation()
    else:
        n_visits = None
        full_area = None
    best_score, best_sequence, best_actions = gcgnrpa(
        environment,
        level,
        n_policies,
        policy,
        n_visits,
        0,
        full_area,
        half_life_divider,
        tau,
        pid,
    )
    environment.disallow_truncation()
    return best_sequence, best_actions, best_score
