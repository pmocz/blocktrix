# blocktrix

A JAX library for solving block tri-diagonal matrix systems.

## Project Structure

```
blocktrix/
├── blocktrix/              # Main package
│   ├── __init__.py         # Public API exports
│   ├── solver_bcyclic.py  # B-cyclic solver implementation
│   └── solver_thomas.py    # Thomas solver implementation
├── tests/
│   └── test_solver.py      # Unit tests
├── pyproject.toml          # Package configuration
└── CLAUDE.md               # This file
```

## Development

### Install in development mode

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
```

### Run tests with coverage

```bash
pytest --cov=blocktrix
```

## Key Components

- `solve_block_tridiagonal(n_blocks, lower, diag, upper, rhs)` - Main solver using block Thomas algorithm
- `build_block_tridiagonal_matrix(lower, diag, upper)` - Utility to construct full matrix
- `random_block_tridiagonal(key, n_blocks, block_size)` - Generate random test systems

## Thomas Algorithm

The serial block Thomas algorithm (block LU decomposition):

1. **Forward sweep**: Eliminate sub-diagonal blocks by computing multipliers and updating diagonal/RHS
2. **Backward substitution**: Solve from last block to first

Complexity: O(n * m^3) where n = number of blocks, m = block size

## B-cyclic Algorithm

The more advanced parallel B-cyclic algorithm. 

* Recursively eliminates alternating block rows, halving the system size at each level.

* Each reduction step factors diagonal blocks and updates neighboring blocks using dense matrix–matrix operations.

* Once a small coarse system is solved, the algorithm back-substitutes to recover the full solution.

Complexity: O(log(n) * m^3) 

## Notes

- The solver is JIT-compiled with JAX for performance
- `n_blocks` must be passed as a static argument (known at compile time)
- Supports both single and multiple right-hand sides
