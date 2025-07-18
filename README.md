# blocktrix

Philip Mocz (2025)

This is simple C++ playground code for solving block tridiagonal linear systems in parallel on CPUs and GPUs.

⚠️ Work-In-Progress ⚠️

- **General:** Works for any block size and number of blocks.
- **Dependencies:** Requires LAPACK (for block LU and solve).
- **Files:**
  - `blocktrix_solver.hpp`/`blocktrix_solver.cpp`: The solver implementation
  - `test_blocktrix.cpp`: Example/test program

## Usage

1. Build and Run Test be running the script

```sh
./install
```

(You may need to adjust LAPACK flags for your system.)
