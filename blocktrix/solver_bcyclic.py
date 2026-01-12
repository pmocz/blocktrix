import math
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

"""
B-cyclic (Block Cyclic Reduction) solver for block tri-diagonal matrix systems.

The B-cyclic algorithm is a parallel algorithm that recursively eliminates
alternating block rows, halving the system size at each level. After solving
the coarse system, back-substitution recovers the full solution.

Complexity: O(log(n) * m^3) where n = number of blocks, m = block size
"""


def _make_eliminate_row(stride, n_blocks):
    """Factory function to create eliminate_row with captured stride."""

    def eliminate_row(carry, k):
        """Eliminate row at position i = (2k+1) * stride."""
        D, L, U, B, tape_Dinv_L_level, tape_Dinv_U_level, tape_Dinv_rhs_level = carry
        i = (2 * k + 1) * stride
        i_prev = i - stride
        i_next = i + stride

        # Compute D_i^{-1} @ L_i, D_i^{-1} @ U_i, D_i^{-1} @ B_i
        D_i = D[i]
        L_i = L[i]
        U_i = U[i]
        B_i = B[i]

        Dinv_L = jax.scipy.linalg.solve(D_i, L_i)
        Dinv_U = jax.scipy.linalg.solve(D_i, U_i)
        Dinv_B = jax.scipy.linalg.solve(D_i, B_i)

        # Store in tape for back-substitution
        tape_Dinv_L_level = tape_Dinv_L_level.at[i].set(Dinv_L)
        tape_Dinv_U_level = tape_Dinv_U_level.at[i].set(Dinv_U)
        tape_Dinv_rhs_level = tape_Dinv_rhs_level.at[i].set(Dinv_B)

        # Update row i_prev
        U_prev = U[i_prev]
        D = D.at[i_prev].set(D[i_prev] - U_prev @ Dinv_L)
        B = B.at[i_prev].set(B[i_prev] - U_prev @ Dinv_B)
        U = U.at[i_prev].set(-U_prev @ Dinv_U)

        # Update row i_next (if it exists)
        def update_next(args):
            D, L, B, Dinv_L, Dinv_U, Dinv_B = args
            L_next = L[i_next]
            D = D.at[i_next].set(D[i_next] - L_next @ Dinv_U)
            B = B.at[i_next].set(B[i_next] - L_next @ Dinv_B)
            L = L.at[i_next].set(-L_next @ Dinv_L)
            return D, L, B

        def no_update(args):
            D, L, B, _, _, _ = args
            return D, L, B

        D, L, B = lax.cond(
            i_next < n_blocks,
            update_next,
            no_update,
            (D, L, B, Dinv_L, Dinv_U, Dinv_B),
        )

        return (
            D,
            L,
            U,
            B,
            tape_Dinv_L_level,
            tape_Dinv_U_level,
            tape_Dinv_rhs_level,
        ), None

    return eliminate_row


def _make_recover_row(stride, n_blocks, tape_Dinv_L, tape_Dinv_U, tape_Dinv_rhs, level):
    """Factory function to create recover_row with captured values."""

    def recover_row(x, k):
        """Recover row at position i = (2k+1) * stride."""
        i = (2 * k + 1) * stride
        i_prev = i - stride
        i_next = i + stride

        # x_i = D_i^{-1} @ (B_i - L_i @ x_{i-s} - U_i @ x_{i+s})
        # The tape stores values for row i (the eliminated row)
        x_i = tape_Dinv_rhs[level, i] - tape_Dinv_L[level, i] @ x[i_prev]

        # Subtract contribution from i_next if it exists
        # Note: tape_Dinv_U[level, i] stores D_i^{-1} @ U_i for the eliminated row i
        x_i = lax.cond(
            i_next < n_blocks,
            lambda args: args[0] - tape_Dinv_U[level, i] @ x[args[1]],
            lambda args: args[0],
            (x_i, i_next),
        )

        x = x.at[i].set(x_i)
        return x, None

    return recover_row


@partial(jax.jit, static_argnums=(0,))
def solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs):
    """
    Solve a block tri-diagonal system using the B-cyclic reduction algorithm.

    Parameters
    ----------
    n_blocks : int
        Number of diagonal blocks. Must be a power of 2.
    lower : jnp.ndarray, shape (n_blocks-1, m, m)
        Sub-diagonal blocks A_1, A_2, ..., A_{n-1}.
    diag : jnp.ndarray, shape (n_blocks, m, m)
        Diagonal blocks B_0, B_1, ..., B_{n-1}.
    upper : jnp.ndarray, shape (n_blocks-1, m, m)
        Super-diagonal blocks C_0, C_1, ..., C_{n-2}.
    rhs : jnp.ndarray, shape (n_blocks, m) or (n_blocks, m, k)
        Right-hand side vector(s).

    Returns
    -------
    x : jnp.ndarray, shape (n_blocks, m) or (n_blocks, m, k)
        Solution vector(s).

    Notes
    -----
    The algorithm proceeds in log2(n_blocks) levels:

    Reduction phase:
        At each level with stride s = 2^level:
        - Eliminate rows at odd multiples of s (s, 3s, 5s, ...)
        - Update neighboring rows at even multiples of s

    Back-substitution:
        Traverse levels in reverse to reconstruct eliminated variables.
    """
    # Validate n_blocks is a power of 2
    if n_blocks < 1 or (n_blocks & (n_blocks - 1)) != 0:
        raise ValueError(f"n_blocks must be a power of 2, got {n_blocks}")

    rhs_shape = rhs.shape
    m = diag.shape[1]

    # Handle both vector and matrix RHS
    if rhs.ndim == 2:
        rhs = rhs[..., None]

    # Handle single block case
    if n_blocks == 1:
        x = jax.scipy.linalg.solve(diag[0], rhs[0])[None, ...]
        if len(rhs_shape) == 2:
            x = x[..., 0]
        return x

    # Pad lower and upper to have n_blocks entries for easier indexing
    zero_block = jnp.zeros((m, m))
    lower_padded = jnp.concatenate([zero_block[None, ...], lower], axis=0)
    upper_padded = jnp.concatenate([upper, zero_block[None, ...]], axis=0)

    # Number of reduction levels
    n_levels = int(math.log2(n_blocks))

    # Storage for tape (for back-substitution)
    tape_Dinv_L = jnp.zeros((n_levels, n_blocks, m, m))
    tape_Dinv_U = jnp.zeros((n_levels, n_blocks, m, m))
    tape_Dinv_rhs = jnp.zeros((n_levels, n_blocks, m, rhs.shape[-1]))

    D = diag
    L = lower_padded
    U = upper_padded
    B = rhs

    # Reduction phase - unrolled at compile time
    for level in range(n_levels):
        stride = 2**level
        n_elim = n_blocks // (2 * stride)

        eliminate_row = _make_eliminate_row(stride, n_blocks)

        tape_Dinv_L_level = tape_Dinv_L[level]
        tape_Dinv_U_level = tape_Dinv_U[level]
        tape_Dinv_rhs_level = tape_Dinv_rhs[level]

        (D, L, U, B, tape_Dinv_L_level, tape_Dinv_U_level, tape_Dinv_rhs_level), _ = (
            lax.scan(
                eliminate_row,
                (D, L, U, B, tape_Dinv_L_level, tape_Dinv_U_level, tape_Dinv_rhs_level),
                jnp.arange(n_elim),
            )
        )

        tape_Dinv_L = tape_Dinv_L.at[level].set(tape_Dinv_L_level)
        tape_Dinv_U = tape_Dinv_U.at[level].set(tape_Dinv_U_level)
        tape_Dinv_rhs = tape_Dinv_rhs.at[level].set(tape_Dinv_rhs_level)

    # Solve the coarse system (only row 0 remains)
    x = jnp.zeros_like(rhs)
    x = x.at[0].set(jax.scipy.linalg.solve(D[0], B[0]))

    # Back-substitution phase - unrolled at compile time
    for level in range(n_levels - 1, -1, -1):
        stride = 2**level
        n_elim = n_blocks // (2 * stride)

        recover_row = _make_recover_row(
            stride, n_blocks, tape_Dinv_L, tape_Dinv_U, tape_Dinv_rhs, level
        )

        x, _ = lax.scan(recover_row, x, jnp.arange(n_elim))

    # Restore original shape
    if len(rhs_shape) == 2:
        x = x[..., 0]

    return x
