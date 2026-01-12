#!/usr/bin/env python

import argparse
import time
import jax
import jax.numpy as jnp

from blocktrix import (
    solve_block_tridiagonal_thomas,
    solve_block_tridiagonal_bcyclic,
    random_block_tridiagonal,
)

"""
Timing comparison of block tri-diagonal solvers.

Compares:
- Thomas algorithm (serial)
- B-cyclic algorithm (parallel-friendly)
- Vanilla dense solve (build full matrix + jnp.linalg.solve)
"""


def time_function(fn, *args, n_runs=3):
    """Time a function, returning the minimum time across runs."""
    # Warmup (JIT compile)
    result = fn(*args)
    result.block_until_ready()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args)
        result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    return min(times), result


def main():

    parser = argparse.ArgumentParser(
        description="Timing comparison of block tri-diagonal solvers"
    )
    parser.add_argument(
        "--n-blocks", type=int, default=1024, help="Number of blocks (default: 1024)"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Size of each block (default: 256)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    n_blocks = args.n_blocks
    block_size = args.block_size

    print("=" * 50)
    print("Block tri-diagonal solver timing comparison")
    print("=" * 50)
    print(f"Number of blocks: {n_blocks}")
    print(f"Block size: {block_size} x {block_size}")
    print(f"Total matrix size: {n_blocks * block_size} x {n_blocks * block_size}")
    print("-" * 50)
    print()

    # Generate random system
    print("Generating random block tri-diagonal system...")
    key = jax.random.PRNGKey(args.seed)
    lower, diag, upper, rhs = random_block_tridiagonal(key, n_blocks, block_size)

    # Force materialization
    jax.block_until_ready([lower, diag, upper, rhs])
    print("Done.\n")

    # Time Thomas solver
    print("Timing Thomas algorithm...")
    thomas_time, x_thomas = time_function(
        lambda: solve_block_tridiagonal_thomas(n_blocks, lower, diag, upper, rhs)
    )
    print(f"  Thomas time: {thomas_time:.4f} s")

    # Time B-cyclic solver
    print("Timing B-cyclic algorithm...")
    bcyclic_time, x_bcyclic = time_function(
        lambda: solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)
    )
    print(f"  B-cyclic time: {bcyclic_time:.4f} s")

    # Check agreement between methods
    error_bt = jnp.max(jnp.abs(x_thomas - x_bcyclic))
    print(f"\n  Max error (Thomas vs B-cyclic): {error_bt:.2e}")
    assert error_bt < 1e-5

    # Summary
    print("\n" + "-" * 50)
    print("Timing summary:")
    print(f"  num blocks: {n_blocks}")
    print(f"  block size: {block_size}")
    print(f"  Thomas:     {thomas_time:.4f} s")
    print(f"  B-cyclic:   {bcyclic_time:.4f} s")
    if bcyclic_time < thomas_time:
        print(f"  B-cyclic is {thomas_time / bcyclic_time:.2f}x faster than Thomas")
    else:
        print(f"  Thomas is {bcyclic_time / thomas_time:.2f}x faster than B-cyclic")
    print("=" * 50)
    print()
    print()


if __name__ == "__main__":
    main()
