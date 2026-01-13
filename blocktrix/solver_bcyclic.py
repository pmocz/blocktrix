import math
import jax
import jax.numpy as jnp
from functools import partial

"""
B-cyclic (Block Cyclic Reduction) solver for block tri-diagonal matrix systems.

The B-cyclic algorithm is a parallel algorithm that recursively eliminates
alternating block rows, halving the system size at each level. After solving
the coarse system, back-substitution recovers the full solution.

Complexity: O(log(n) * m^3) where n = number of blocks, m = block size
"""


def _eliminate_level(D, L, U, B, stride, n_blocks, n_elim):
    """
    Eliminate all odd-multiple rows at a given level using batched LU solves.

    At each level with stride s, we eliminate rows at positions i = (2k+1)*s
    for k = 0, 1, ..., n_elim-1. All these LU solves are independent and
    can be computed in parallel using vmap.
    """
    m = D.shape[1]

    # Compute indices for rows to eliminate
    ks = jnp.arange(n_elim)
    elim_indices = (2 * ks + 1) * stride  # rows being eliminated
    prev_indices = elim_indices - stride  # rows before (always valid)
    next_indices = elim_indices + stride  # rows after (may be out of bounds)

    # Gather matrices for eliminated rows
    D_elim = D[elim_indices]  # (n_elim, m, m)
    L_elim = L[elim_indices]  # (n_elim, m, m)
    U_elim = U[elim_indices]  # (n_elim, m, m)
    B_elim = B[elim_indices]  # (n_elim, m, k)

    # Batched LU solves: D_i^{-1} @ [L_i, U_i, B_i] for all eliminated rows
    # jax.scipy.linalg.solve supports batching on leading dimensions
    Dinv_L = jax.scipy.linalg.solve(D_elim, L_elim)  # (n_elim, m, m)
    Dinv_U = jax.scipy.linalg.solve(D_elim, U_elim)  # (n_elim, m, m)
    Dinv_B = jax.scipy.linalg.solve(D_elim, B_elim)  # (n_elim, m, k)

    # Store tape values using scatter
    tape_Dinv_L_level = jnp.zeros((n_blocks, m, m)).at[elim_indices].set(Dinv_L)
    tape_Dinv_U_level = jnp.zeros((n_blocks, m, m)).at[elim_indices].set(Dinv_U)
    tape_Dinv_rhs_level = (
        jnp.zeros((n_blocks, m, B.shape[-1])).at[elim_indices].set(Dinv_B)
    )

    # Update previous rows (i_prev = i - stride): always valid
    U_prev = U[prev_indices]  # (n_elim, m, m)
    D_prev_update = D[prev_indices] - jnp.einsum("bij,bjk->bik", U_prev, Dinv_L)
    B_prev_update = B[prev_indices] - jnp.einsum("bij,bjk->bik", U_prev, Dinv_B)
    U_prev_update = -jnp.einsum("bij,bjk->bik", U_prev, Dinv_U)

    D = D.at[prev_indices].set(D_prev_update)
    B = B.at[prev_indices].set(B_prev_update)
    U = U.at[prev_indices].set(U_prev_update)

    # Update next rows (i_next = i + stride): only if in bounds
    # Create mask for valid next indices
    valid_next = next_indices < n_blocks

    # Gather L_next, but use zeros for out-of-bounds indices
    safe_next_indices = jnp.where(valid_next, next_indices, 0)
    L_next = L[safe_next_indices]  # (n_elim, m, m)

    # Compute updates (will be masked)
    D_next_vals = D[safe_next_indices] - jnp.einsum("bij,bjk->bik", L_next, Dinv_U)
    B_next_vals = B[safe_next_indices] - jnp.einsum("bij,bjk->bik", L_next, Dinv_B)
    L_next_vals = -jnp.einsum("bij,bjk->bik", L_next, Dinv_L)

    # Only update where valid_next is True
    # Use where to mask updates
    D_next_update = jnp.where(
        valid_next[:, None, None], D_next_vals, D[safe_next_indices]
    )
    B_next_update = jnp.where(
        valid_next[:, None, None], B_next_vals, B[safe_next_indices]
    )
    L_next_update = jnp.where(
        valid_next[:, None, None], L_next_vals, L[safe_next_indices]
    )

    D = D.at[safe_next_indices].set(D_next_update)
    B = B.at[safe_next_indices].set(B_next_update)
    L = L.at[safe_next_indices].set(L_next_update)

    return D, L, U, B, tape_Dinv_L_level, tape_Dinv_U_level, tape_Dinv_rhs_level


def _recover_level(
    x, tape_Dinv_L, tape_Dinv_U, tape_Dinv_rhs, stride, n_blocks, n_elim, level
):
    """
    Recover all eliminated rows at a given level using batched operations.

    At each level with stride s, we recover rows at positions i = (2k+1)*s
    for k = 0, 1, ..., n_elim-1.
    """
    # Compute indices
    ks = jnp.arange(n_elim)
    elim_indices = (2 * ks + 1) * stride
    prev_indices = elim_indices - stride
    next_indices = elim_indices + stride

    # Gather tape values for eliminated rows
    Dinv_rhs = tape_Dinv_rhs[level, elim_indices]  # (n_elim, m, k)
    Dinv_L = tape_Dinv_L[level, elim_indices]  # (n_elim, m, m)
    Dinv_U = tape_Dinv_U[level, elim_indices]  # (n_elim, m, m)

    # Gather x values for previous rows (always valid)
    x_prev = x[prev_indices]  # (n_elim, m, k)

    # x_i = Dinv_rhs - Dinv_L @ x_prev - Dinv_U @ x_next (if valid)
    x_elim = Dinv_rhs - jnp.einsum("bij,bjk->bik", Dinv_L, x_prev)

    # Handle next rows (may be out of bounds)
    valid_next = next_indices < n_blocks
    safe_next_indices = jnp.where(valid_next, next_indices, 0)
    x_next = x[safe_next_indices]  # (n_elim, m, k)

    # Subtract contribution from next rows where valid
    next_contrib = jnp.einsum("bij,bjk->bik", Dinv_U, x_next)
    next_contrib = jnp.where(valid_next[:, None, None], next_contrib, 0.0)
    x_elim = x_elim - next_contrib

    # Scatter back to x
    x = x.at[elim_indices].set(x_elim)

    return x


@partial(jax.jit, static_argnums=(0,))
def _solve_block_tridiagonal_bcyclic_pow2(n_blocks, lower, diag, upper, rhs):
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

    # Reduction phase - unrolled at compile time with batched LU solves
    for level in range(n_levels):
        stride = 2**level
        n_elim = n_blocks // (2 * stride)

        D, L, U, B, tape_Dinv_L_level, tape_Dinv_U_level, tape_Dinv_rhs_level = (
            _eliminate_level(D, L, U, B, stride, n_blocks, n_elim)
        )

        tape_Dinv_L = tape_Dinv_L.at[level].set(tape_Dinv_L_level)
        tape_Dinv_U = tape_Dinv_U.at[level].set(tape_Dinv_U_level)
        tape_Dinv_rhs = tape_Dinv_rhs.at[level].set(tape_Dinv_rhs_level)

    # Solve the coarse system (only row 0 remains)
    x = jnp.zeros_like(rhs)
    x = x.at[0].set(jax.scipy.linalg.solve(D[0], B[0]))

    # Back-substitution phase - unrolled at compile time with batched operations
    for level in range(n_levels - 1, -1, -1):
        stride = 2**level
        n_elim = n_blocks // (2 * stride)

        x = _recover_level(
            x, tape_Dinv_L, tape_Dinv_U, tape_Dinv_rhs, stride, n_blocks, n_elim, level
        )

    # Restore original shape
    if len(rhs_shape) == 2:
        x = x[..., 0]

    return x


@partial(jax.jit, static_argnums=(0,))
def solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs):
    """
    Solve a block tri-diagonal system using the B-cyclic reduction algorithm.

    This function handles any number of blocks by padding to the next power of 2
    when necessary.

    Parameters
    ----------
    n_blocks : int
        Number of diagonal blocks (any positive integer).
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
    For n_blocks that are not a power of 2, the system is padded with identity
    diagonal blocks and zero off-diagonal blocks. The padded system is solved
    using the B-cyclic algorithm, and the solution is extracted for the
    original blocks.
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")

    # Check if n_blocks is a power of 2
    is_power_of_2 = (n_blocks & (n_blocks - 1)) == 0

    if is_power_of_2:
        return _solve_block_tridiagonal_bcyclic_pow2(n_blocks, lower, diag, upper, rhs)

    # Pad to next power of 2
    n_padded = 1 << (n_blocks - 1).bit_length()
    n_pad = n_padded - n_blocks
    m = diag.shape[1]

    # Pad diagonal with identity blocks
    I_blocks = jnp.tile(jnp.eye(m)[None, ...], (n_pad, 1, 1))
    diag_padded = jnp.concatenate([diag, I_blocks], axis=0)

    # Pad off-diagonals with zero blocks
    zero_blocks = jnp.zeros((n_pad, m, m))
    lower_padded = jnp.concatenate([lower, zero_blocks], axis=0)
    upper_padded = jnp.concatenate([upper, zero_blocks], axis=0)

    # Pad RHS with zeros
    rhs_pad_shape = (n_pad,) + rhs.shape[1:]
    rhs_padded = jnp.concatenate([rhs, jnp.zeros(rhs_pad_shape)], axis=0)

    # Solve padded system
    x_padded = _solve_block_tridiagonal_bcyclic_pow2(
        n_padded, lower_padded, diag_padded, upper_padded, rhs_padded
    )

    # Extract original solution
    return x_padded[:n_blocks]
