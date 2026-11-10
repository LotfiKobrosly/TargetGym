from classes.environment import EnvironmentWrapper


def random_walk(environment: EnvironmentWrapper):
    done = False
    environment.reset()
    while not environment.is_final():
        environment.step(environment.sample_random_action())
    return environment.sequence, environment.actions, environment.score
