import glob
import numpy as np
import matplotlib.pyplot as plt

"""
plot timing results from timing.py

Usage:
    python plot.py
"""


def read_timing_results(filename):
    """
    Read and parse the slurm*.out file
    to get all the n_blocks and block_size combinations
    and the corresponding times for Thomas and B-cyclic.

    Example timing summary block:

    Timing summary:
      num blocks: 512
      block size: 128
      Thomas:     0.3040 s
      B-cyclic:   0.2317 s
    """
    n_blocks_list = []
    block_size_list = []
    thomas_times = []
    bcyclic_times = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if "Timing summary:" in line:
            n_blocks = int(lines[i + 1].split(":")[1].strip())
            block_size = int(lines[i + 2].split(":")[1].strip())
            thomas_time = float(lines[i + 3].split(":")[1].strip().split()[0])
            bcyclic_time = float(lines[i + 4].split(":")[1].strip().split()[0])

            n_blocks_list.append(n_blocks)
            block_size_list.append(block_size)
            thomas_times.append(thomas_time)
            bcyclic_times.append(bcyclic_time)

            i += 5  # Move past this block
        else:
            i += 1  # Move to next line

    return (
        np.array(n_blocks_list),
        np.array(block_size_list),
        np.array(thomas_times),
        np.array(bcyclic_times),
    )


def main():
    # Find the largest slurm-gpu*.out file
    slurm_gpu_files = glob.glob("slurm-gpu*.out")
    if not slurm_gpu_files:
        print("No slurm-gpu output files found.")
        return
    slurm_cpu_files = glob.glob("slurm-cpu*.out")
    if not slurm_cpu_files:
        print("No slurm-cpu output files found.")
        return

    gpu_file = max(slurm_gpu_files)
    cpu_file = max(slurm_cpu_files)
    print(f"Reading timing results from {gpu_file} and {cpu_file}")

    (
        n_blocks,
        block_sizes,
        thomas_times,
        bcyclic_times,
    ) = read_timing_results(gpu_file)

    (
        n_blocks_cpu,
        block_sizes_cpu,
        thomas_times_cpu,
        bcyclic_times_cpu,
    ) = read_timing_results(cpu_file)

    assert (n_blocks == n_blocks_cpu).all()
    assert (block_sizes == block_sizes_cpu).all()

    # unique_n_blocks = np.unique(n_blocks)
    unique_block_sizes = np.unique(block_sizes)[::-1]  # Reverse to plot largest first

    # Plot Timing
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use colormap for different block sizes
    colors = plt.cm.tab10(np.arange(len(unique_block_sizes)))

    for i, bs in enumerate(unique_block_sizes):
        mask = block_sizes == bs
        # Thomas (cpu)
        ax.plot(
            n_blocks[mask],
            thomas_times_cpu[mask],
            marker="o",
            linestyle=":",
            color=colors[i],
            linewidth=1,
            markersize=6,
            markeredgewidth=1,
            markerfacecolor="none",
            label=f"Thomas - CPU (block-size={bs})",
        )
        # Thomas (gpu)
        ax.plot(
            n_blocks[mask],
            thomas_times[mask],
            marker="o",
            linestyle="--",
            color=colors[i],
            linewidth=2,
            markersize=6,
            markeredgewidth=2,
            markerfacecolor="none",
            label=f"Thomas - GPU (block-size={bs})",
        )
        # B-cyclic (gpu)
        ax.plot(
            n_blocks[mask],
            bcyclic_times[mask],
            marker="s",
            linestyle="-",
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=f"B-cyclic - GPU (block-size={bs})",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks([256, 512, 1024, 2048])
    ax.set_xticklabels([256, 512, 1024, 2048])
    ax.tick_params(axis="both", which="both", direction="in", length=6)
    ax.yaxis.set_ticks_position("both")
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel("number of blocks", fontsize=12)
    ax.set_ylabel("time (s)", fontsize=12)
    ax.set_title("Timing Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    plt.tight_layout()
    plt.savefig("timing.png", dpi=150)
    plt.close()
    print("Saved plot as timing.png")

    # Plot Speedup (B-cyclic GPU vs Thomas GPU and CPU)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, bs in enumerate(unique_block_sizes):
        mask = block_sizes == bs
        speedup_gpu = thomas_times[mask] / bcyclic_times[mask]
        speedup_cpu = thomas_times_cpu[mask] / bcyclic_times[mask]
        ax.plot(
            n_blocks[mask],
            speedup_gpu,
            marker="s",
            linestyle="-",
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=f"vs Thomas GPU (block-size={bs})",
        )
        ax.plot(
            n_blocks[mask],
            speedup_cpu,
            marker="o",
            linestyle="--",
            color=colors[i],
            linewidth=1,
            markersize=6,
            markeredgewidth=1,
            markerfacecolor="none",
            label=f"vs Thomas CPU (block-size={bs})",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048])
    ax.set_xticklabels([256, 512, 1024, 2048])
    ax.tick_params(axis="both", which="both", direction="in", length=6)
    ax.yaxis.set_ticks_position("both")
    ax.set_ylim(0, 120)
    ax.set_xlabel("number of blocks", fontsize=12)
    ax.set_ylabel("speedup", fontsize=12)
    ax.set_title(
        "Speedup of B-cyclic (GPU) over Thomas (GPU,CPU)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    plt.tight_layout()
    plt.savefig("speedup.png", dpi=150)
    plt.close()
    print("Saved plot as speedup.png")


if __name__ == "__main__":
    main()
