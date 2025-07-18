#ifndef BTHOMAS_SOLVER_HPP
#define BTHOMAS_SOLVER_HPP

#include <cstddef>
#include <vector>

// Forward declaration for LAPACK routines
extern "C" {
void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv,
             int *info);
void dgetrs_(const char *trans, const int *n, const int *nrhs, const double *a,
             const int *lda, const int *ipiv, double *b, const int *ldb,
             int *info);
}

class BThomasSolver {
  public:
    // n_blocks: number of block rows/columns
    // block_size: size of each block (assumed square)
    BThomasSolver(size_t n_blocks, size_t block_size);

    // Set the block tridiagonal system
    // Each block is stored in a flat vector<double> in row-major order
    // lower_blocks: (n_blocks-1)*block_size*block_size
    // diag_blocks: n_blocks*block_size*block_size
    // upper_blocks: (n_blocks-1)*block_size*block_size
    void set_blocks(const std::vector<double> &lower_blocks,
                    const std::vector<double> &diag_blocks,
                    const std::vector<double> &upper_blocks);

    // Solve the system A x = b, where b is a vector of size n_blocks*block_size
    // Returns the solution x in the same format
    std::vector<double> solve(const std::vector<double> &b);
    std::vector<double> solve_lapack(const std::vector<double> &b);

  private:
    size_t n_blocks_;
    size_t block_size_;
    // Flat storage for all blocks
    std::vector<double> lower_blocks_;
    std::vector<double> diag_blocks_;
    std::vector<double> upper_blocks_;

    // Helper functions
    void invert_block(std::vector<double> &block);
};

#endif // BTHOMAS_SOLVER_HPP