import warnings
import numpy as np
import pandas as pd
import jax
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from classes.environment import EnvironmentWrapper
from solvers.residual_solvers.gcnrpa import GcNRPA_Policy
from utils.models import get_environment
from utils.seeds_generator import jax_keys_generator


def evaluate_policy(environment, parameters, policy, horizon, jax_keys):
    observation, state = environment.reset(jax_keys["reset"], parameters)
    total = 0
    for _ in range(horizon):
        action = policy(np.asarray(observation))
        observation, state, reward, terminated, truncated, _ = gym_environment.step(
            jax_keys["step"], state, action, parameters
        )
        total += float(reward)
        if terminated or truncated:
            break
    return total


if __name__ == "__main__":

    # Global parameters

    ## Random seeds
    n_keys = 4
    jax_keys = dict()
    n_seeds = 10
    training_seeds = [0, 1, 2]
    evaluation_seeds = list(range(0, 10))

    ## Environment setup
    problem = "plane"
    gym_environment, parameters = get_environment(problem)
    horizon = 10

    ## Initialize environment
    jax_keys["main"], jax_keys["reset"], jax_keys["action"], jax_keys["step"] = (
        jax_keys_generator(0, n_keys)
    )
    environment = EnvironmentWrapper(
        gym_environment, parameters, horizon, jax_keys=jax_keys
    )

    ## GcNRPA initialization
    level = 1
    n_policies = 10
    policy = GcNRPA_Policy(environment, level, n_policies)

    # Training
    for seed in training_seeds:
        print("Training for seed =", seed)
        jax_keys["main"], jax_keys["reset"], jax_keys["action"], jax_keys["step"] = (
            jax_keys_generator(seed, n_keys)
        )
        policy.reset(jax_keys)
        policy.train_full_gcnrpa_policy()

    results = {"PID": list(), "Residual GcNRPA": list()}
    # Evaluation
    for seed in evaluation_seeds:
        print("\nEvaluating for seed =", seed)
        ## Random seeds
        jax_keys["main"], jax_keys["reset"], jax_keys["action"], jax_keys["step"] = (
            jax_keys_generator(seed, n_keys)
        )

        ## PID
        gym_environment.reset(jax_keys["reset"])
        pid = gym_environment.make_pid()
        pid_score = evaluate_policy(gym_environment, parameters, pid, horizon, jax_keys)
        results["PID"].append(pid_score)
        # print("PID:", pid_score)

        ## Residual GcNRPA
        policy.reset(jax_keys)
        gcnrpa_score = evaluate_policy(
            gym_environment, parameters, policy, horizon, jax_keys
        )
        results["Residual GcNRPA"].append(gcnrpa_score)
        # print("Residual GcNRPA:", gcnrpa_score)

        print("Ratio: ", gcnrpa_score / pid_score)
