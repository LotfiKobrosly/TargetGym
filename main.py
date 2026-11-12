import warnings
import time
import numpy as np
import pandas as pd
import jax
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from target_gym.registry import REGISTRY

from classes.environment import EnvironmentWrapper
from solvers.residual_solvers.gcnrpa import Residual_GcNRPA_Policy
from utils.models import get_environment
from utils.seeds_generator import jax_keys_generator


def evaluate_policy(environment, parameters, policy, horizon, jax_keys):

    # Reset and initialize total reward
    observation, state = environment.reset(jax_keys["reset"], parameters)
    total = 0

    # Run
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
    problems_list = [
        "four_tank",
        "cement_kiln",
        "glass_furnace",
        "wind_turbine",
        "plane",
        "cstr",
    ]
    horizons_list = [1200, 700, 1600, 400, 280, 150]

    # Iterating over problems
    for problem_id, problem in enumerate(problems_list):
        print("\nCurrent problem:", problem)
        gym_environment, parameters = get_environment(problem)
        horizon = horizons_list[problem_id]

        ## Initialize environment
        jax_keys["main"], jax_keys["reset"], jax_keys["action"], jax_keys["step"] = (
            jax_keys_generator(0, n_keys)
        )
        environment = EnvironmentWrapper(
            gym_environment, parameters, horizon, jax_keys=jax_keys
        )

        ## GcNRPA initialization
        hyperparameters = {
            1: [200, 400, 1000],
            2: [20, 40, 50],
            3: [10, 20, 40],
        }

        residual_policies = {
            "Level: "
            + str(level)
            + ", policies:"
            + str(n_policies): Residual_GcNRPA_Policy(environment, level, n_policies)
            for level, n_policies_list in hyperparameters.items()
            for n_policies in n_policies_list
        }

        training_times = {
            policy_name: list() for policy_name in residual_policies.keys()
        }
        # Training
        for seed in training_seeds:
            print("\nTraining for seed =", seed)
            (
                jax_keys["main"],
                jax_keys["reset"],
                jax_keys["action"],
                jax_keys["step"],
            ) = jax_keys_generator(seed, n_keys)
            for policy_name, policy in residual_policies.items():
                policy.reset(jax_keys)
                start_time = time.time()
                policy.train()
                training_times[policy_name].append(time.time() - start_time)

        results = {"PID": list()}
        for policy_name in residual_policies.keys():

            results[policy_name] = list()
            results["Ratio for " + policy_name] = list()

        # Evaluation
        for seed in evaluation_seeds:
            print("\nEvaluating for seed =", seed)

            ## Random seeds
            (
                jax_keys["main"],
                jax_keys["reset"],
                jax_keys["action"],
                jax_keys["step"],
            ) = jax_keys_generator(seed, n_keys)

            ## PID
            gym_environment.reset(jax_keys["reset"])
            pid = gym_environment.make_pid()
            pid_score = evaluate_policy(
                gym_environment, parameters, pid, horizon, jax_keys
            )
            results["PID"].append(pid_score)

            ## Residual GcNRPA
            for policy_name, policy in residual_policies.items():
                policy.reset(jax_keys)
                score = evaluate_policy(
                    gym_environment, parameters, policy, horizon, jax_keys
                )
                results[policy_name].append(score)

        for policy_name in residual_policies.keys():
            results["Ratio for " + policy_name] = (
                np.array(results[policy_name]) / np.array(results["PID"]) - 1
            )
        dataframe = pd.DataFrame.from_dict(results)
        dataframe.index.name = "Seed"

        filename = "Results_" + problem + "_horizon_" + str(horizon) + ".xlsx"
        dataframe.to_excel(filename)
