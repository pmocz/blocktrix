import jax
import jax.numpy as jnp

"""
Utility functions for block tri-diagonal systems.
"""


def build_block_tridiagonal_matrix(lower, diag, upper):
    """
    Build the full block tri-diagonal matrix from components.

    Parameters
    ----------
    lower : jnp.ndarray, shape (n_blocks-1, m, m)
        Sub-diagonal blocks.
    diag : jnp.ndarray, shape (n_blocks, m, m)
        Diagonal blocks.
    upper : jnp.ndarray, shape (n_blocks-1, m, m)
        Super-diagonal blocks.

    Returns
    -------
    M : jnp.ndarray, shape (n_blocks*m, n_blocks*m)
        The full matrix.
    """
    n_blocks = diag.shape[0]
    m = diag.shape[1]
    N = n_blocks * m

    M = jnp.zeros((N, N))

    for i in range(n_blocks):
        M = M.at[i * m : (i + 1) * m, i * m : (i + 1) * m].set(diag[i])

    for i in range(n_blocks - 1):
        M = M.at[i * m : (i + 1) * m, (i + 1) * m : (i + 2) * m].set(upper[i])

    for i in range(n_blocks - 1):
        M = M.at[(i + 1) * m : (i + 2) * m, i * m : (i + 1) * m].set(lower[i])

    return M


def random_block_tridiagonal(key, n_blocks, block_size, diag_dominant=True):
    """
    Generate a random block tri-diagonal system.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    n_blocks : int
        Number of blocks.
    block_size : int
        Size of each block (m x m).
    diag_dominant : bool, default=True
        If True, make the system diagonally dominant for stability.

    Returns
    -------
    lower : jnp.ndarray, shape (n_blocks-1, block_size, block_size)
        Sub-diagonal blocks.
    diag : jnp.ndarray, shape (n_blocks, block_size, block_size)
        Diagonal blocks.
    upper : jnp.ndarray, shape (n_blocks-1, block_size, block_size)
        Super-diagonal blocks.
    rhs : jnp.ndarray, shape (n_blocks, block_size)
        Right-hand side vector.
    """
    keys = jax.random.split(key, 4)

    lower = jax.random.normal(keys[0], (n_blocks - 1, block_size, block_size))
    diag = jax.random.normal(keys[1], (n_blocks, block_size, block_size))
    upper = jax.random.normal(keys[2], (n_blocks - 1, block_size, block_size))
    rhs = jax.random.normal(keys[3], (n_blocks, block_size))

    if diag_dominant:
        for i in range(n_blocks):
            row_sum = jnp.zeros(block_size)
            if i > 0:
                row_sum = row_sum + jnp.sum(jnp.abs(lower[i - 1]), axis=1)
            if i < n_blocks - 1:
                row_sum = row_sum + jnp.sum(jnp.abs(upper[i]), axis=1)
            diag = diag.at[i].set(diag[i] + jnp.diag(row_sum + 1.0))

    return lower, diag, upper, rhs
