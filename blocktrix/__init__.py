"""
blocktrix: A JAX library for solving block tri-diagonal matrix systems.
"""

from blocktrix.solver import (
    solve_block_tridiagonal,
    build_block_tridiagonal_matrix,
    random_block_tridiagonal,
)

__version__ = "0.1.0"
__all__ = [
    "solve_block_tridiagonal",
    "build_block_tridiagonal_matrix",
    "random_block_tridiagonal",
]
