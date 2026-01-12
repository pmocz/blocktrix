# blocktrix

[![Repo Status][status-badge]][status-link]
[![PyPI Version Status][pypi-badge]][pypi-link]
[![Test Status][workflow-test-badge]][workflow-test-link]
[![Ruff][ruff-badge]][ruff-link]
[![License][license-badge]][license-link]
[![Software DOI][software-doi-badge]][software-doi-link]

[status-link]:         https://www.repostatus.org/#active
[status-badge]:        https://www.repostatus.org/badges/latest/active.svg
[pypi-link]:           https://pypi.org/project/blocktrix
[pypi-badge]:          https://img.shields.io/pypi/v/blocktrix?label=PyPI&logo=pypi
[workflow-test-link]:  https://github.com/pmocz/blocktrix/actions/workflows/test-package.yml
[workflow-test-badge]: https://github.com/pmocz/blocktrix/actions/workflows/test-package.yml/badge.svg?event=push
[ruff-link]:           https://github.com/astral-sh/ruff
[ruff-badge]:          https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[license-link]:        https://opensource.org/licenses/MIT
[license-badge]:       https://img.shields.io/badge/license-MIT-blue.svg
[software-doi-link]:   https://doi.org/10.5281/zenodo.18226420
[software-doi-badge]:  https://zenodo.org/badge/1132904104.svg

A JAX library for efficiently solving block tri-diagonal matrix systems on GPUs.

Author: [Philip Mocz (@pmocz)](https://github.com/pmocz/)

Implements both a block [**Thomas**](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm) (serial) solver and a block cyclic reduction [**B-cyclic**](https://ui.adsabs.harvard.edu/abs/2010JCoPh.229.6392H) (parallel) solver.

⚠️ **Warning: Work in Progress** ⚠️

This library is still under active development and is not guaranteed to work at this point XXX.


## Installation

```bash
pip install blocktrix
```

## Usage


```python
import jax
from blocktrix import solve_block_tridiagonal_bcyclic, random_block_tridiagonal

# Generate a random test system
key = jax.random.PRNGKey(42)
n_blocks, block_size = 8, 4

lower, diag, upper, rhs = random_block_tridiagonal(key, n_blocks, block_size)

# Solve the system
x = solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)
```

![Quickstart](examples/quickstart/solution.png)


## It's fast!

![Timing comparison](examples/timing/timing.png)

![Speedup](examples/timing/speedup.png)


## Links

* [Code repository](https://github.com/pmocz/blocktrix)
* [Documentation](https://github.com/pmocz/blocktrix)


## Cite this repository

If you use this software, please cite it as below.

```bibtex
@software{Mocz_Blocktrix_2026,
   author = {Mocz, Philip},
      doi = {https://doi.org/10.5281/zenodo.18226420},
    month = jan,
    title = {{Blocktrix}},
      url = {https://github.com/pmocz/blocktrix},
  version = {0.0.4},
     year = {2026}
}
```
