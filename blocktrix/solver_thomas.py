import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

"""
Core Thomas solver (simple/serial) for block tri-diagonal matrix systems.

A block tri-diagonal matrix has the form:
    [B0  C0  0   0   ... 0  ]
    [A1  B1  C1  0   ... 0  ]
    [0   A2  B2  C2  ... 0  ]
    [...                    ]
    [0   ... 0   A_{n-1}  B_{n-1}]

where A_i, B_i, C_i are square blocks of size (m x m).
"""


@partial(jax.jit, static_argnums=(0,))
def solve_block_tridiagonal_thomas(n_blocks, lower, diag, upper, rhs):
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
