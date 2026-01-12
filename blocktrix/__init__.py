from importlib.metadata import version, PackageNotFoundError

"""
blocktrix: A JAX library for solving block tri-diagonal matrix systems.
"""

from blocktrix.solver_thomas import (
    solve_block_tridiagonal,
    build_block_tridiagonal_matrix,
    random_block_tridiagonal,
)

try:
    __version__ = version("blocktrix")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "solve_block_tridiagonal",
    "build_block_tridiagonal_matrix",
    "random_block_tridiagonal",
]
