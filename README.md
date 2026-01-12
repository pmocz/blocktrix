# blocktrix

A JAX library for efficiently solving block tri-diagonal matrix systems

Author: [Philip Mocz (@pmocz)](https://github.com/pmocz/)

## Installation

```bash
pip install blocktrix
```

## Usage

XXX

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

## It's fast!

XXX

## Links

* [Code repository](https://github.com/pmocz/blocktrix)
* [Documentation](XXX) XXX


## Cite this repository

If you use this software, please cite it as below.

XXX