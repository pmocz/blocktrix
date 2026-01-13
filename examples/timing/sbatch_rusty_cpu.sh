#!/usr/bin/bash
#SBATCH --job-name=timing-cpu
#SBATCH --output=slurm-cpu-%j.out
#SBATCH --error=slurm-cpu-%j.err
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=00-00:20

module purge
module load python/3.12.9

export PYTHONUNBUFFERED=TRUE

source $VENVDIR/blocktrix-venv/bin/activate

srun python timing.py --n-blocks 256 --block-size 64
srun python timing.py --n-blocks 256 --block-size 128
srun python timing.py --n-blocks 256 --block-size 256
srun python timing.py --n-blocks 256 --block-size 512

srun python timing.py --n-blocks 512 --block-size 64
srun python timing.py --n-blocks 512 --block-size 128
srun python timing.py --n-blocks 512 --block-size 256
srun python timing.py --n-blocks 512 --block-size 512

srun python timing.py --n-blocks 1024 --block-size 64
srun python timing.py --n-blocks 1024 --block-size 128
srun python timing.py --n-blocks 1024 --block-size 256
srun python timing.py --n-blocks 1024 --block-size 512

srun python timing.py --n-blocks 2048 --block-size 64
srun python timing.py --n-blocks 2048 --block-size 128
srun python timing.py --n-blocks 2048 --block-size 256
srun python timing.py --n-blocks 2048 --block-size 512