"""
Core Thomas solver for block tri-diagonal matrix systems.

A block tri-diagonal matrix has the form:
    [B0  C0  0   0   ... 0  ]
    [A1  B1  C1  0   ... 0  ]
    [0   A2  B2  C2  ... 0  ]
    [...                    ]
    [0   ... 0   A_{n-1}  B_{n-1}]

where A_i, B_i, C_i are square blocks of size (m x m).
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


@partial(jax.jit, static_argnums=(0,))
def solve_block_tridiagonal(n_blocks, lower, diag, upper, rhs):
    """
    Solve a block tri-diagonal system using the block Thomas algorithm.

    Parameters
    ----------
    n_blocks : int
        Number of diagonal blocks (static, used for loop unrolling).
    lower : jnp.ndarray, shape (n_blocks-1, m, m)
        Sub-diagonal blocks A_1, A_2, ..., A_{n-1}.
    diag : jnp.ndarray, shape (n_blocks, m, m)
        Diagonal blocks B_0, B_1, ..., B_{n-1}.
    upper : jnp.ndarray, shape (n_blocks-1, m, m)
        Super-diagonal blocks C_0, C_1, ..., C_{n-2}.
    rhs : jnp.ndarray, shape (n_blocks, m) or (n_blocks, m, k)
        Right-hand side vector(s) d.

    Returns
    -------
    x : jnp.ndarray, shape (n_blocks, m) or (n_blocks, m, k)
        Solution vector(s).

    Notes
    -----
    The algorithm performs block LU factorization:

    Forward sweep (modify diagonal and rhs):
        For i = 1, ..., n-1:
            w_i = A_i @ inv(B_{i-1})
            B_i = B_i - w_i @ C_{i-1}
            d_i = d_i - w_i @ d_{i-1}

    Backward substitution:
        x_{n-1} = inv(B_{n-1}) @ d_{n-1}
        For i = n-2, ..., 0:
            x_i = inv(B_i) @ (d_i - C_i @ x_{i+1})
    """
    rhs_shape = rhs.shape

    # Handle both vector and matrix RHS
    if rhs.ndim == 2:
        rhs = rhs[..., None]

    # Handle single block case (just a direct solve)
    if n_blocks == 1:
        x = jax.scipy.linalg.solve(diag[0], rhs[0])[None, ...]
        if len(rhs_shape) == 2:
            x = x[..., 0]
        return x

    def forward_step(carry, i):
        diag_mod, rhs_mod = carry

        # w = A_i @ inv(B_{i-1})
        w = jax.scipy.linalg.solve(diag_mod[i - 1].T, lower[i - 1].T).T

        # Update diagonal: B_i = B_i - w @ C_{i-1}
        new_diag_i = diag_mod[i] - w @ upper[i - 1]
        diag_mod = diag_mod.at[i].set(new_diag_i)

        # Update RHS: d_i = d_i - w @ d_{i-1}
        new_rhs_i = rhs_mod[i] - w @ rhs_mod[i - 1]
        rhs_mod = rhs_mod.at[i].set(new_rhs_i)

        return (diag_mod, rhs_mod), None

    (diag_mod, rhs_mod), _ = lax.scan(
        forward_step, (diag, rhs), jnp.arange(1, n_blocks)
    )

    def backward_step(carry, i):
        x = carry

        # x_i = inv(B_i) @ (d_i - C_i @ x_{i+1})
        residual = rhs_mod[i] - upper[i] @ x[i + 1]
        new_x_i = jax.scipy.linalg.solve(diag_mod[i], residual)
        x = x.at[i].set(new_x_i)

        return x, None

    # Initialize solution array
    x = jnp.zeros_like(rhs_mod)

    # Solve last block
    x = x.at[n_blocks - 1].set(
        jax.scipy.linalg.solve(diag_mod[n_blocks - 1], rhs_mod[n_blocks - 1])
    )

    # Backward sweep
    x, _ = lax.scan(backward_step, x, jnp.arange(n_blocks - 2, -1, -1))

    # Restore original shape
    if len(rhs_shape) == 2:
        x = x[..., 0]

    return x


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
