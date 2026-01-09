# blocktrix

A JAX library for solving block tri-diagonal matrix systems using the block Thomas algorithm.

## Installation

```bash
pip install -e .
```

## Usage

```python
import jax
from blocktrix import solve_block_tridiagonal, random_block_tridiagonal

# Generate a random test system
key = jax.random.PRNGKey(42)
n_blocks, block_size = 5, 3

lower, diag, upper, rhs = random_block_tridiagonal(key, n_blocks, block_size)

# Solve the system
x = solve_block_tridiagonal(n_blocks, lower, diag, upper, rhs)
```

## License

MIT
