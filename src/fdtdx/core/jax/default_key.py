import jax

_DEFAULT_KEY_SEED = 0


def default_key(key: jax.Array | None) -> jax.Array:
    """Return *key* unchanged or create a deterministic key from a fixed seed.

    Use this at the top of any public function that accepts an optional
    ``key`` argument so callers can omit it for deterministic workflows
    while still being able to pass an explicit key when needed.

    Args:
        key: A JAX PRNG key, or ``None``.

    Returns:
        The original key if one was supplied, otherwise
        ``jax.random.PRNGKey(0)``.
    """
    if key is None:
        return jax.random.PRNGKey(_DEFAULT_KEY_SEED)
    return key
