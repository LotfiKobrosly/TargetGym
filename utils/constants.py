import numpy as np
import jax

RANDOM_SEED = 0
RANDOM_STATE = np.random.default_rng(seed=RANDOM_SEED)
JAX_KEY = jax.random.PRNGKey(RANDOM_SEED)
MAIN_KEY, RESET_KEY, ACTION_KEY, STEP_KEY = jax.random.split(JAX_KEY, 4)

HALF_LIFE_DIVIDER = 10

LEARNING_RATE = 0.1

RELEVANCE_THRESHOLD = 0.3
