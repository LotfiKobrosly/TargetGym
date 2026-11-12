import jax


def jax_keys_generator(random_seed: int, n_keys: int):
    jax_key = jax.random.PRNGKey(random_seed)
    return jax.random.split(jax_key, 4)
