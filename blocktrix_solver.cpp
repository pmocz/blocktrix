#include "blocktrix_solver.hpp"
#include <cmath> // For std::abs
#include <cstring>
#include <stdexcept>

BlocktrixSolver::BlocktrixSolver(size_t n_blocks, size_t block_size)
    : n_blocks_(n_blocks), block_size_(block_size) {}

void BlocktrixSolver::set_blocks(const std::vector<double> &lower_blocks,
                                 const std::vector<double> &diag_blocks,
                                 const std::vector<double> &upper_blocks) {
    size_t bs = block_size_;
    if (diag_blocks.size() != n_blocks_ * bs * bs)
        throw std::invalid_argument("diag_blocks size mismatch");
    if (lower_blocks.size() != (n_blocks_ - 1) * bs * bs)
        throw std::invalid_argument("lower_blocks size mismatch");
    if (upper_blocks.size() != (n_blocks_ - 1) * bs * bs)
        throw std::invalid_argument("upper_blocks size mismatch");
    lower_blocks_ = lower_blocks;
    diag_blocks_ = diag_blocks;
    upper_blocks_ = upper_blocks;
}

// Helper: invert a small square matrix in place (row-major)
void BlocktrixSolver::invert_block(std::vector<double> &block) {
    size_t n = block_size_;
    std::vector<double> inv(n * n, 0.0);
    for (size_t i = 0; i < n; ++i)
        inv[i * n + i] = 1.0;
    for (size_t i = 0; i < n; ++i) {
        double pivot = block[i * n + i];
        if (std::abs(pivot) < 1e-14)
            throw std::runtime_error("Singular block in invert_block");
        for (size_t j = 0; j < n; ++j) {
            block[i * n + j] /= pivot;
            inv[i * n + j] /= pivot;
        }
        for (size_t k = 0; k < n; ++k) {
            if (k == i)
                continue;
            double factor = block[k * n + i];
            for (size_t j = 0; j < n; ++j) {
                block[k * n + j] -= factor * block[i * n + j];
                inv[k * n + j] -= factor * inv[i * n + j];
            }
        }
    }
    block = inv;
}

// Helper: transpose a square matrix (row-major <-> column-major)
static void transpose_square(const std::vector<double> &in,
                             std::vector<double> &out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            out[j * n + i] = in[i * n + j];
}

std::vector<double> BlocktrixSolver::solve(const std::vector<double> &b) {
    if (b.size() != n_blocks_ * block_size_)
        throw std::invalid_argument("RHS size mismatch");
    size_t N = n_blocks_;
    size_t bs = block_size_;
    // Copy blocks and rhs
    std::vector<double> a = lower_blocks_;
    std::vector<double> bdiag = diag_blocks_;
    std::vector<double> c = upper_blocks_;
    std::vector<double> d = b;
    // Forward elimination
    for (size_t i = 0; i < N - 1; ++i) {
        // Invert bdiag[i]
        std::vector<double> bdiag_inv(bs * bs);
        for (size_t j = 0; j < bs * bs; ++j)
            bdiag_inv[j] = bdiag[i * bs * bs + j];
        invert_block(bdiag_inv);
        // Compute m = a[i] * bdiag_inv
        std::vector<double> m(bs * bs, 0.0);
        for (size_t row = 0; row < bs; ++row) {
            for (size_t col = 0; col < bs; ++col) {
                for (size_t k = 0; k < bs; ++k) {
                    m[row * bs + col] +=
                        a[i * bs * bs + row * bs + k] * bdiag_inv[k * bs + col];
                }
            }
        }
        // Update bdiag[i+1] = bdiag[i+1] - m * c[i]
        std::vector<double> mc(bs * bs, 0.0);
        for (size_t row = 0; row < bs; ++row) {
            for (size_t col = 0; col < bs; ++col) {
                for (size_t k = 0; k < bs; ++k) {
                    mc[row * bs + col] +=
                        m[row * bs + k] * c[i * bs * bs + k * bs + col];
                }
            }
        }
        for (size_t j = 0; j < bs * bs; ++j) {
            bdiag[(i + 1) * bs * bs + j] -= mc[j];
        }
        // Update d_{i+1} = d_{i+1} - m * d_i
        std::vector<double> md(bs, 0.0);
        for (size_t row = 0; row < bs; ++row) {
            for (size_t k = 0; k < bs; ++k) {
                md[row] += m[row * bs + k] * d[i * bs + k];
            }
        }
        for (size_t j = 0; j < bs; ++j) {
            d[(i + 1) * bs + j] -= md[j];
        }
    }
    // Backward substitution
    std::vector<double> x(N * bs, 0.0);
    // Solve last block
    std::vector<double> bdiag_inv(bs * bs);
    for (size_t j = 0; j < bs * bs; ++j)
        bdiag_inv[j] = bdiag[(N - 1) * bs * bs + j];
    invert_block(bdiag_inv);
    for (size_t i = 0; i < bs; ++i) {
        for (size_t k = 0; k < bs; ++k) {
            x[(N - 1) * bs + i] += bdiag_inv[i * bs + k] * d[(N - 1) * bs + k];
        }
    }
    // Backward
    for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
        std::vector<double> rhs(bs, 0.0);
        for (size_t row = 0; row < bs; ++row) {
            rhs[row] = d[i * bs + row];
            for (size_t k = 0; k < bs; ++k) {
                rhs[row] -= c[i * bs * bs + row * bs + k] * x[(i + 1) * bs + k];
            }
        }
        std::vector<double> bdiag_inv(bs * bs);
        for (size_t j = 0; j < bs * bs; ++j)
            bdiag_inv[j] = bdiag[i * bs * bs + j];
        invert_block(bdiag_inv);
        for (size_t row = 0; row < bs; ++row) {
            for (size_t k = 0; k < bs; ++k) {
                x[i * bs + row] += bdiag_inv[row * bs + k] * rhs[k];
            }
        }
    }
    return x;
}

std::vector<double>
BlocktrixSolver::solve_lapack(const std::vector<double> &b) {
    if (b.size() != n_blocks_ * block_size_)
        throw std::invalid_argument("RHS size mismatch");
    size_t N = n_blocks_;
    size_t bs = block_size_;
    std::vector<double> a = lower_blocks_;
    std::vector<double> bdiag = diag_blocks_;
    std::vector<double> c = upper_blocks_;
    std::vector<double> d = b;
    std::vector<std::vector<int>> pivots(N, std::vector<int>(bs));
    int info = 0;
    for (size_t i = 0; i < N - 1; ++i) {
        int n = static_cast<int>(bs);
        dgetrf_(&n, &n, &bdiag[i * bs * bs], &n, pivots[i].data(), &info);
        if (info != 0)
            throw std::runtime_error("LU factorization failed on block " +
                                     std::to_string(i));
        std::vector<double> cwork_row(bs * bs);
        for (size_t j = 0; j < bs * bs; ++j)
            cwork_row[j] = c[i * bs * bs + j];
        std::vector<double> cwork_col(bs * bs);
        transpose_square(cwork_row, cwork_col, bs);
        char trans = 'T';
        int nrhs = n;
        dgetrs_(&trans, &n, &nrhs, &bdiag[i * bs * bs], &n, pivots[i].data(),
                cwork_col.data(), &n, &info);
        if (info != 0)
            throw std::runtime_error(
                "dgetrs_ failed for upper block on block " + std::to_string(i));
        transpose_square(cwork_col, cwork_row, bs);
        std::vector<double> ac(bs * bs, 0.0);
        for (size_t row = 0; row < bs; ++row) {
            for (size_t col = 0; col < bs; ++col) {
                for (size_t k = 0; k < bs; ++k) {
                    ac[row * bs + col] +=
                        a[i * bs * bs + row * bs + k] * cwork_row[k * bs + col];
                }
            }
        }
        for (size_t j = 0; j < bs * bs; ++j) {
            bdiag[(i + 1) * bs * bs + j] -= ac[j];
        }
        std::vector<double> dwork_blk(bs);
        for (size_t k = 0; k < bs; ++k)
            dwork_blk[k] = d[i * bs + k];
        int one = 1;
        dgetrs_(&trans, &n, &one, &bdiag[i * bs * bs], &n, pivots[i].data(),
                dwork_blk.data(), &n, &info);
        if (info != 0)
            throw std::runtime_error("dgetrs_ failed for rhs on block " +
                                     std::to_string(i));
        std::vector<double> ad(bs, 0.0);
        for (size_t row = 0; row < bs; ++row) {
            for (size_t k = 0; k < bs; ++k) {
                ad[row] += a[i * bs * bs + row * bs + k] * dwork_blk[k];
            }
        }
        for (size_t j = 0; j < bs; ++j) {
            d[(i + 1) * bs + j] -= ad[j];
        }
    }
    int n = static_cast<int>(bs);
    dgetrf_(&n, &n, &bdiag[(N - 1) * bs * bs], &n, pivots[N - 1].data(), &info);
    if (info != 0)
        throw std::runtime_error("LU factorization failed on block " +
                                 std::to_string(N - 1));
    std::vector<double> x(N * bs, 0.0);
    std::vector<double> dwork_blk(bs);
    for (size_t k = 0; k < bs; ++k)
        dwork_blk[k] = d[(N - 1) * bs + k];
    int one = 1;
    char trans = 'T';
    dgetrs_(&trans, &n, &one, &bdiag[(N - 1) * bs * bs], &n,
            pivots[N - 1].data(), dwork_blk.data(), &n, &info);
    if (info != 0)
        throw std::runtime_error("dgetrs_ failed for rhs on last block");
    for (size_t i = 0; i < bs; ++i)
        x[(N - 1) * bs + i] = dwork_blk[i];
    for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
        std::vector<double> rhs(bs, 0.0);
        for (size_t row = 0; row < bs; ++row) {
            rhs[row] = d[i * bs + row];
            for (size_t k = 0; k < bs; ++k) {
                rhs[row] -= c[i * bs * bs + row * bs + k] * x[(i + 1) * bs + k];
            }
        }
        for (size_t k = 0; k < bs; ++k)
            dwork_blk[k] = rhs[k];
        dgetrs_(&trans, &n, &one, &bdiag[i * bs * bs], &n, pivots[i].data(),
                dwork_blk.data(), &n, &info);
        if (info != 0)
            throw std::runtime_error("dgetrs_ failed for rhs on block " +
                                     std::to_string(i));
        for (size_t row = 0; row < bs; ++row)
            x[i * bs + row] = dwork_blk[row];
    }
    return x;
}