import numpy as np
import matplotlib.pyplot as plt
import jax

from utils.constants import RESET_KEY, ACTION_KEY, STEP_KEY, RANDOM_SEED
from utils.models import code


class EnvironmentWrapper:
    def __init__(
        self, environment, parameters, simulation_horizon: float,
    ):
        self.environment = environment
        self.parameters = parameters
        # Initialize the environment
        observation, info = environment.reset(key=RESET_KEY)
        self.current_state = code(observation)
        self.sequence = [self.current_state]
        self.best_sequence = [self.current_state]
        self.actions = list()
        self.best_actions = list()
        self.states_actions = dict()
        self.current_timestamp = 0
        self.horizon = simulation_horizon
        self.best_score = -np.inf
        self.done = False
        self.score = 0
        self.cumulative_reward = 0
        self.truncate = False

    def is_final(self):
        return self.done or self.current_timestamp >= self.horizon

    def allow_truncation(self):
        self.truncate = True

    def disallow_truncation(self):
        self.truncate = False

    def reset(self):
        observation, state = self.environment.reset(key=RESET_KEY)
        observation = code(observation)
        if self.truncate:
            observation = self.truncate_observation(observation)
            self.environment.obs = observation[:]
        self.current_state = observation
        self.current_hidden_state = state
        self.sequence = [self.current_state]
        self.best_sequence = [self.current_state]
        self.actions = list()
        self.best_actions = list()
        self.states_actions = dict()
        self.current_timestamp = 0
        self.best_score = -np.inf
        self.done = False
        self.score = 0

    def truncate_observation(self, observation):
        new_observation = tuple()
        for component_index, component in enumerate(observation):
            if component > 2:
                new_observation = new_observation, 1
            if component < -2:
                new_observation = new_observation, -1
        return new_observation

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        action = action.flatten()
        # print(action.shape)
        # print("Here: ", type(self.current_state))
        observation, new_state, reward, terminated, info = self.environment.step_env(
            STEP_KEY,
            self.current_hidden_state,
            action,
            # params=self.parameters,
        )
        observation = code(observation)
        if self.truncate:
            observation = self.truncate_observation(observation)
            self.environment.obs = observation[:]
        self.actions.append(action)
        self.current_timestamp += 1
        self.done = terminated
        self.sequence.append(observation)
        self.current_state = observation
        self.current_hidden_state = new_state
        self.score += reward

    def sample_random_action(self, n_samples):
        keys = jax.random.split(ACTION_KEY, n_samples)
        if n_samples == 1:
            return self.environment.action_space().sample(key=ACTION_KEY)
        else:
            return [
                self.environment.action_space().sample(key=key)
                for key in keys
            ]

    def close(self):
        self.environment.close()

    def get_pid(self):
        return self.environment.make_pid()
