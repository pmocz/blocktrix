import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from blocktrix import (
    solve_block_tridiagonal_thomas,
    solve_block_tridiagonal_bcyclic,
    build_block_tridiagonal_matrix,
    random_block_tridiagonal,
)

"""
Basic example demonstrating the blocktrix library.

Solves a block tri-diagonal system: M @ x = b
"""

# Enable double precision
jax.config.update("jax_enable_x64", True)


def main():
    # System parameters
    n_blocks = 8  # Number of blocks (must be power of 2 for bcyclic)
    block_size = 4  # Size of each block

    # Generate a random block tri-diagonal system
    key = jax.random.PRNGKey(42)
    lower, diag, upper, rhs = random_block_tridiagonal(
        key, n_blocks, block_size, diag_dominant=False
    )

    print("Solving block tri-diagonal system:")
    print(f"  n_blocks: {n_blocks}")
    print(f"  block_size: {block_size}")
    print(f"  Total system size: {n_blocks * block_size} x {n_blocks * block_size}")

    # Solve using Thomas algorithm (serial)
    x_thomas = solve_block_tridiagonal_thomas(n_blocks, lower, diag, upper, rhs)

    # Solve using B-cyclic algorithm (parallel)
    x_bcyclic = solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)

    # Solve using full matrix solver (for verification)
    M = build_block_tridiagonal_matrix(lower, diag, upper)
    b = rhs.flatten()
    x_full = jnp.linalg.solve(M, b)

    # Verify solutions by computing residuals
    residual_full = jnp.linalg.norm(M @ x_full.flatten() - b)
    residual_thomas = jnp.linalg.norm(M @ x_thomas.flatten() - b)
    residual_bcyclic = jnp.linalg.norm(M @ x_bcyclic.flatten() - b)

    # Solve using full matrix solver (for verification)
    print()
    print(f"Residual (Full):    {residual_full:.2e}")
    print(f"Residual (Thomas):  {residual_thomas:.2e}")
    print(f"Residual (B-cyclic): {residual_bcyclic:.2e}")
    print(f"Solutions match (t/b): {jnp.allclose(x_thomas, x_bcyclic)}")
    print(f"Solutions match (t/f): {jnp.allclose(x_thomas.flatten(), x_full)}")
    print(f"Solutions match (b/f): {jnp.allclose(x_bcyclic.flatten(), x_full)}")

    # Make a plot of the matrix and the solution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    # plt.spy(M, markersize=1)
    vmax = jnp.max(jnp.abs(M))
    plt.imshow(M, cmap="bwr", aspect=1, vmin=-vmax, vmax=vmax)
    plt.title("Block Tri-Diagonal Matrix M")
    plt.subplot(1, 2, 2)
    plt.plot(x_full, "-", linewidth=2, label="Full")
    plt.plot(
        x_thomas.flatten(),
        "--",
        linewidth=2,
        marker="o",
        markevery=2,
        markerfacecolor="none",
        label="Thomas",
    )
    plt.plot(
        x_bcyclic.flatten(),
        "-.",
        linewidth=2,
        marker="s",
        markevery=2,
        label="B-cyclic",
    )
    plt.xlabel("index")
    plt.ylabel("value")
    plt.legend()
    plt.title("Solution Vector x")
    plt.tight_layout()
    plt.savefig("solution.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
