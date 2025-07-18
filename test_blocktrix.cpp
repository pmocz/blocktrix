#include "bthomas_solver.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Helper to multiply block tridiagonal matrix by a vector (flat block storage)
void block_tridiag_matvec(size_t n_blocks, size_t block_size,
                          const std::vector<double> &lower_blocks,
                          const std::vector<double> &diag_blocks,
                          const std::vector<double> &upper_blocks,
                          const std::vector<double> &x,
                          std::vector<double> &y) {
    y.assign(n_blocks * block_size, 0.0);
    for (size_t i = 0; i < n_blocks; ++i) {
        // Diagonal block
        for (size_t row = 0; row < block_size; ++row) {
            for (size_t col = 0; col < block_size; ++col) {
                y[i * block_size + row] +=
                    diag_blocks[i * block_size * block_size + row * block_size +
                                col] *
                    x[i * block_size + col];
            }
        }
        // Lower block
        if (i > 0) {
            for (size_t row = 0; row < block_size; ++row) {
                for (size_t col = 0; col < block_size; ++col) {
                    y[i * block_size + row] +=
                        lower_blocks[(i - 1) * block_size * block_size +
                                     row * block_size + col] *
                        x[(i - 1) * block_size + col];
                }
            }
        }
        // Upper block
        if (i < n_blocks - 1) {
            for (size_t row = 0; row < block_size; ++row) {
                for (size_t col = 0; col < block_size; ++col) {
                    y[i * block_size + row] +=
                        upper_blocks[i * block_size * block_size +
                                     row * block_size + col] *
                        x[(i + 1) * block_size + col];
                }
            }
        }
    }
}

int main() {
    constexpr size_t n_blocks = 10000;
    constexpr size_t block_size = 40;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Generate random blocks (flat storage)
    std::vector<double> lower_blocks((n_blocks - 1) * block_size * block_size);
    std::vector<double> diag_blocks(n_blocks * block_size * block_size);
    std::vector<double> upper_blocks((n_blocks - 1) * block_size * block_size);
    for (size_t i = 0; i < n_blocks - 1; ++i) {
        for (size_t j = 0; j < block_size * block_size; ++j) {
            lower_blocks[i * block_size * block_size + j] = dist(rng);
            upper_blocks[i * block_size * block_size + j] = dist(rng);
        }
    }
    for (size_t i = 0; i < n_blocks; ++i) {
        for (size_t j = 0; j < block_size * block_size; ++j) {
            diag_blocks[i * block_size * block_size + j] = dist(rng);
        }
        for (size_t d = 0; d < block_size; ++d) {
            diag_blocks[i * block_size * block_size + d * block_size + d] +=
                block_size;
        }
    }

    // Generate a random solution x_true
    std::vector<double> x_true(n_blocks * block_size);
    for (auto &v : x_true)
        v = dist(rng);

    // Compute b = A x_true
    std::vector<double> b;
    block_tridiag_matvec(n_blocks, block_size, lower_blocks, diag_blocks,
                         upper_blocks, x_true, b);

    // Solve Ax = b with basic solver
    BThomasSolver solver(n_blocks, block_size);
    solver.set_blocks(lower_blocks, diag_blocks, upper_blocks);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> x = solver.solve(b);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Solve time: (simple serial block Thomas): " << duration.count() / 1000.0 << " ms"
              << std::endl;

    // Compute error
    double max_err = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        max_err = std::max(max_err, std::abs(x[i] - x_true[i]));
    }
    std::cout << "Max error: (simple serial block Thomas): " << max_err << std::endl;
    assert(max_err < 1e-8);

    // LAPACK-based solver
    solver.set_blocks(lower_blocks, diag_blocks,
                      upper_blocks); // reset blocks in case they were modified
    auto start_lapack = std::chrono::high_resolution_clock::now();
    std::vector<double> x_lapack = solver.solve_lapack(b);
    auto end_lapack = std::chrono::high_resolution_clock::now();
    auto duration_lapack =
        std::chrono::duration_cast<std::chrono::microseconds>(end_lapack -
                                                              start_lapack);
    std::cout << "Solve time (LAPACK serial block Thomas): "
              << duration_lapack.count() / 1000.0 << " ms" << std::endl;
    double max_err_lapack = 0.0;
    for (size_t i = 0; i < x_lapack.size(); ++i) {
        max_err_lapack =
            std::max(max_err_lapack, std::abs(x_lapack[i] - x_true[i]));
    }
    std::cout << "Max error (LAPACK serial block Thomas): " << max_err_lapack
              << std::endl;
    assert(max_err_lapack < 1e-8);
    return 0;
}