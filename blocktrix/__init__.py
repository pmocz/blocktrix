from importlib.metadata import version, PackageNotFoundError

from blocktrix.solver_bcyclic import solve_block_tridiagonal_bcyclic
from blocktrix.solver_thomas import solve_block_tridiagonal_thomas
from blocktrix.utils import random_block_tridiagonal, build_block_tridiagonal_matrix

"""
blocktrix: A JAX library for solving block tri-diagonal matrix systems.
"""

try:
    __version__ = version("blocktrix")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "solve_block_tridiagonal_bcyclic",
    "solve_block_tridiagonal_thomas",
    "build_block_tridiagonal_matrix",
    "random_block_tridiagonal",
]
