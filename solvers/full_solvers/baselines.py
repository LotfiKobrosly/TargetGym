import numpy as np
from stable_baselines3 import PPO, A2C, DDPG, SAC
from classes.environment import EnvironmentWrapper


observation_types = {
    "array": "MlpPolicy",
    "dict": "MultiInputPolicy",
}


def get_and_train_ppo(
    environment: EnvironmentWrapper,
    n_steps_learning: int = int(3e4),
    learning_rate: float = 0.001,
    *args,
    **kwargs,
):
    return PPO(
        "MlpPolicy", environment.environment, verbose=1, learning_rate=learning_rate
    ).learn(total_timesteps=n_steps_learning, log_interval=n_steps_learning // 100)


def get_and_train_ddpg(
    environment: EnvironmentWrapper,
    n_steps_learning: int = int(3e4),
    *args,
    **kwargs,
):
    return DDPG("MlpPolicy", environment.environment, verbose=1).learn(
        total_timesteps=n_steps_learning, log_interval=n_steps_learning // 100
    )


def get_and_train_a2c(
    environment: EnvironmentWrapper,
    n_steps_learning: int = int(3e4),
    *args,
    **kwargs,
):
    return A2C("MlpPolicy", environment.environment, verbose=1).learn(
        total_timesteps=n_steps_learning, log_interval=n_steps_learning // 100
    )


def get_and_train_sac(
    environment: EnvironmentWrapper,
    n_steps_learning: int = int(3e4),
    *args,
    **kwargs,
):
    return SAC("MlpPolicy", environment.environment, verbose=1).learn(
        total_timesteps=n_steps_learning, log_interval=n_steps_learning // 100
    )


def get_and_train_baseline(
    environment: EnvironmentWrapper,
    model_type: str,
    *args,
    **kwargs,
):
    environment.reset()
    models_dict = {
        "A2C": get_and_train_a2c,
        "PPO": get_and_train_ppo,
        "DDPG": get_and_train_ddpg,
        "SAC": get_and_train_sac,
    }
    assert model_type in list(models_dict.keys()), (
        "Chosen model " + model_type + " undefined or non implemented"
    )
    return models_dict[model_type](environment, **kwargs)


def run_baseline(environment: EnvironmentWrapper, model: dict()):
    observation, info = environment.environment.reset()
    while not environment.is_final():
        action, observation = model.predict(environment.current_state)
        # action = np.array([action]).reshape(-1)
        environment.step(action)

    return environment.sequence, environment.actions, environment.score
