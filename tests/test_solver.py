import jax
import jax.numpy as jnp

from blocktrix import (
    solve_block_tridiagonal_thomas,
    solve_block_tridiagonal_bcyclic,
    build_block_tridiagonal_matrix,
    random_block_tridiagonal,
)

"""Tests for blocktrix solver."""


class TestSolveBlockTridiagonalThomas:
    """Tests for the block tri-diagonal solver."""

    def test_small_system(self):
        """Test a small 3-block system."""
        key = jax.random.PRNGKey(42)
        n_blocks, block_size = 3, 2

        lower, diag, upper, rhs = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        x = solve_block_tridiagonal_thomas(n_blocks, lower, diag, upper, rhs)

        # Verify solution
        M = build_block_tridiagonal_matrix(lower, diag, upper)
        residual = jnp.linalg.norm(M @ x.flatten() - rhs.flatten())
        assert residual < 1e-5

    def test_larger_system(self):
        """Test a larger system."""
        key = jax.random.PRNGKey(123)
        n_blocks, block_size = 10, 4

        lower, diag, upper, rhs = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        x = solve_block_tridiagonal_thomas(n_blocks, lower, diag, upper, rhs)

        M = build_block_tridiagonal_matrix(lower, diag, upper)
        relative_error = jnp.linalg.norm(
            M @ x.flatten() - rhs.flatten()
        ) / jnp.linalg.norm(rhs.flatten())
        assert relative_error < 1e-5

    def test_multiple_rhs(self):
        """Test solving with multiple right-hand sides."""
        key = jax.random.PRNGKey(456)
        n_blocks, block_size = 5, 3
        n_rhs = 4

        lower, diag, upper, _ = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        key2 = jax.random.PRNGKey(789)
        rhs_multi = jax.random.normal(key2, (n_blocks, block_size, n_rhs))

        x_multi = solve_block_tridiagonal_thomas(
            n_blocks, lower, diag, upper, rhs_multi
        )

        assert x_multi.shape == (n_blocks, block_size, n_rhs)

        # Verify each RHS solution
        M = build_block_tridiagonal_matrix(lower, diag, upper)
        for k in range(n_rhs):
            x_k = x_multi[..., k].flatten()
            rhs_k = rhs_multi[..., k].flatten()
            residual = jnp.linalg.norm(M @ x_k - rhs_k)
            assert residual < 1e-5

    def test_compare_to_direct_solve(self):
        """Compare block Thomas to direct solve."""
        key = jax.random.PRNGKey(999)
        n_blocks, block_size = 6, 3

        lower, diag, upper, rhs = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        x_block = solve_block_tridiagonal_thomas(n_blocks, lower, diag, upper, rhs)

        M = build_block_tridiagonal_matrix(lower, diag, upper)
        x_direct = jnp.linalg.solve(M, rhs.flatten())

        diff = jnp.linalg.norm(x_block.flatten() - x_direct)
        assert diff < 1e-5

    def test_single_block(self):
        """Test edge case with single block (just a matrix solve)."""
        key = jax.random.PRNGKey(111)
        n_blocks, block_size = 1, 4

        keys = jax.random.split(key, 2)
        diag = jax.random.normal(keys[0], (1, block_size, block_size))
        diag = diag + jnp.eye(block_size) * 5  # Make well-conditioned
        rhs = jax.random.normal(keys[1], (1, block_size))

        lower = jnp.zeros((0, block_size, block_size))
        upper = jnp.zeros((0, block_size, block_size))

        x = solve_block_tridiagonal_thomas(n_blocks, lower, diag, upper, rhs)

        # Direct solve for comparison
        x_direct = jnp.linalg.solve(diag[0], rhs[0])
        diff = jnp.linalg.norm(x[0] - x_direct)
        assert diff < 1e-6


class TestSolveBlockTridiagonalBcyclic:
    """Tests for the B-cyclic block tri-diagonal solver."""

    def test_small_system(self):
        """Test a small 4-block system (power of 2)."""
        key = jax.random.PRNGKey(42)
        n_blocks, block_size = 4, 2

        lower, diag, upper, rhs = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        x = solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)

        # Verify solution
        M = build_block_tridiagonal_matrix(lower, diag, upper)
        residual = jnp.linalg.norm(M @ x.flatten() - rhs.flatten())
        assert residual < 1e-5

    def test_larger_system(self):
        """Test a larger system (power of 2)."""
        key = jax.random.PRNGKey(123)
        n_blocks, block_size = 16, 4

        lower, diag, upper, rhs = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        x = solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)

        M = build_block_tridiagonal_matrix(lower, diag, upper)
        relative_error = jnp.linalg.norm(
            M @ x.flatten() - rhs.flatten()
        ) / jnp.linalg.norm(rhs.flatten())
        assert relative_error < 1e-5

    def test_multiple_rhs(self):
        """Test solving with multiple right-hand sides."""
        key = jax.random.PRNGKey(456)
        n_blocks, block_size = 8, 3
        n_rhs = 4

        lower, diag, upper, _ = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        key2 = jax.random.PRNGKey(789)
        rhs_multi = jax.random.normal(key2, (n_blocks, block_size, n_rhs))

        x_multi = solve_block_tridiagonal_bcyclic(
            n_blocks, lower, diag, upper, rhs_multi
        )

        assert x_multi.shape == (n_blocks, block_size, n_rhs)

        # Verify each RHS solution
        M = build_block_tridiagonal_matrix(lower, diag, upper)
        for k in range(n_rhs):
            x_k = x_multi[..., k].flatten()
            rhs_k = rhs_multi[..., k].flatten()
            residual = jnp.linalg.norm(M @ x_k - rhs_k)
            assert residual < 1e-5

    def test_compare_to_thomas(self):
        """Compare B-cyclic to Thomas solver."""
        key = jax.random.PRNGKey(999)
        n_blocks, block_size = 8, 3

        lower, diag, upper, rhs = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        x_bcyclic = solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)
        x_thomas = solve_block_tridiagonal_thomas(n_blocks, lower, diag, upper, rhs)

        assert jnp.allclose(x_bcyclic, x_thomas, rtol=1e-5)

    def test_single_block(self):
        """Test edge case with single block."""
        key = jax.random.PRNGKey(111)
        n_blocks, block_size = 1, 4

        keys = jax.random.split(key, 2)
        diag = jax.random.normal(keys[0], (1, block_size, block_size))
        diag = diag + jnp.eye(block_size) * 5  # Make well-conditioned
        rhs = jax.random.normal(keys[1], (1, block_size))

        lower = jnp.zeros((0, block_size, block_size))
        upper = jnp.zeros((0, block_size, block_size))

        x = solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)

        # Direct solve for comparison
        x_direct = jnp.linalg.solve(diag[0], rhs[0])
        diff = jnp.linalg.norm(x[0] - x_direct)
        assert diff < 1e-6

    def test_two_blocks(self):
        """Test with two blocks (simplest non-trivial case)."""
        key = jax.random.PRNGKey(222)
        n_blocks, block_size = 2, 3

        lower, diag, upper, rhs = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        x = solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)

        M = build_block_tridiagonal_matrix(lower, diag, upper)
        residual = jnp.linalg.norm(M @ x.flatten() - rhs.flatten())
        assert residual < 1e-5

    def test_various_power_of_two_sizes(self):
        """Test various power-of-2 system sizes."""
        key = jax.random.PRNGKey(333)
        block_size = 2

        for n_blocks in [2, 4, 8, 16, 32]:
            lower, diag, upper, rhs = random_block_tridiagonal(
                key, n_blocks, block_size, diag_dominant=True
            )

            x = solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)

            M = build_block_tridiagonal_matrix(lower, diag, upper)
            residual = jnp.linalg.norm(M @ x.flatten() - rhs.flatten())
            assert residual < 1e-4, f"Failed for n_blocks={n_blocks}"

    def test_non_power_of_two_sizes(self):
        """Test non-power-of-2 system sizes (uses padding)."""
        key = jax.random.PRNGKey(444)
        block_size = 2

        for n_blocks in [3, 5, 6, 7, 9, 10, 15, 17, 31, 33]:
            lower, diag, upper, rhs = random_block_tridiagonal(
                key, n_blocks, block_size, diag_dominant=True
            )

            x = solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)

            M = build_block_tridiagonal_matrix(lower, diag, upper)
            relative_error = jnp.linalg.norm(
                M @ x.flatten() - rhs.flatten()
            ) / jnp.linalg.norm(rhs.flatten())
            assert relative_error < 1e-4, f"Failed for n_blocks={n_blocks}"

    def test_non_power_of_two_compare_to_thomas(self):
        """Compare B-cyclic with padding to Thomas solver for non-power-of-2."""
        key = jax.random.PRNGKey(555)
        n_blocks, block_size = 11, 3

        lower, diag, upper, rhs = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        x_bcyclic = solve_block_tridiagonal_bcyclic(n_blocks, lower, diag, upper, rhs)
        x_thomas = solve_block_tridiagonal_thomas(n_blocks, lower, diag, upper, rhs)

        assert jnp.allclose(x_bcyclic, x_thomas, rtol=1e-5)

    def test_non_power_of_two_multiple_rhs(self):
        """Test non-power-of-2 with multiple right-hand sides."""
        key = jax.random.PRNGKey(666)
        n_blocks, block_size = 7, 3
        n_rhs = 4

        lower, diag, upper, _ = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        key2 = jax.random.PRNGKey(777)
        rhs_multi = jax.random.normal(key2, (n_blocks, block_size, n_rhs))

        x_multi = solve_block_tridiagonal_bcyclic(
            n_blocks, lower, diag, upper, rhs_multi
        )

        assert x_multi.shape == (n_blocks, block_size, n_rhs)

        # Verify each RHS solution
        M = build_block_tridiagonal_matrix(lower, diag, upper)
        for k in range(n_rhs):
            x_k = x_multi[..., k].flatten()
            rhs_k = rhs_multi[..., k].flatten()
            residual = jnp.linalg.norm(M @ x_k - rhs_k)
            assert residual < 1e-5


class TestBuildBlockTridiagonalMatrix:
    """Tests for building the full matrix."""

    def test_structure(self):
        """Verify the matrix has correct tri-diagonal block structure."""
        n_blocks, block_size = 4, 2

        lower = jnp.ones((n_blocks - 1, block_size, block_size)) * 1
        diag = jnp.ones((n_blocks, block_size, block_size)) * 2
        upper = jnp.ones((n_blocks - 1, block_size, block_size)) * 3

        M = build_block_tridiagonal_matrix(lower, diag, upper)

        N = n_blocks * block_size
        assert M.shape == (N, N)

        # Check diagonal blocks
        for i in range(n_blocks):
            block = M[
                i * block_size : (i + 1) * block_size,
                i * block_size : (i + 1) * block_size,
            ]
            assert jnp.allclose(block, 2.0)

        # Check upper diagonal blocks
        for i in range(n_blocks - 1):
            block = M[
                i * block_size : (i + 1) * block_size,
                (i + 1) * block_size : (i + 2) * block_size,
            ]
            assert jnp.allclose(block, 3.0)

        # Check lower diagonal blocks
        for i in range(n_blocks - 1):
            block = M[
                (i + 1) * block_size : (i + 2) * block_size,
                i * block_size : (i + 1) * block_size,
            ]
            assert jnp.allclose(block, 1.0)


class TestRandomBlockTridiagonal:
    """Tests for random system generation."""

    def test_shapes(self):
        """Verify output shapes."""
        key = jax.random.PRNGKey(42)
        n_blocks, block_size = 5, 3

        lower, diag, upper, rhs = random_block_tridiagonal(key, n_blocks, block_size)

        assert lower.shape == (n_blocks - 1, block_size, block_size)
        assert diag.shape == (n_blocks, block_size, block_size)
        assert upper.shape == (n_blocks - 1, block_size, block_size)
        assert rhs.shape == (n_blocks, block_size)

    def test_diagonal_dominance(self):
        """Verify diagonal dominance when requested."""
        key = jax.random.PRNGKey(42)
        n_blocks, block_size = 5, 3

        lower, diag, upper, _ = random_block_tridiagonal(
            key, n_blocks, block_size, diag_dominant=True
        )

        # Build full matrix and check condition number is reasonable
        M = build_block_tridiagonal_matrix(lower, diag, upper)
        cond = jnp.linalg.cond(M)
        assert cond < 1e6  # Well-conditioned
