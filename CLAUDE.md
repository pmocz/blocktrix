# blocktrix

A JAX library for solving block tri-diagonal matrix systems.

## Project Structure

```
blocktrix/
├── blocktrix/          # Main package
│   ├── __init__.py     # Public API exports
│   └── solver.py       # Core solver implementation
├── tests/
│   └── test_solver.py  # Unit tests
├── pyproject.toml      # Package configuration
└── CLAUDE.md           # This file
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

## Algorithm

Uses the block Thomas algorithm (block LU decomposition):

1. **Forward sweep**: Eliminate sub-diagonal blocks by computing multipliers and updating diagonal/RHS
2. **Backward substitution**: Solve from last block to first

Complexity: O(n * m^3) where n = number of blocks, m = block size

## Notes

- The solver is JIT-compiled with JAX for performance
- `n_blocks` must be passed as a static argument (known at compile time)
- Supports both single and multiple right-hand sides
