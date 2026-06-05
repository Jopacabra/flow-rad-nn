#!/bin/bash
#SBATCH --job-name=rad_data_gen
#SBATCH --array=0-19   # Each batch has a specific seed -- deterministic samples for given n-points & array val
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     # Must match --n-workers below
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/sobol_%A_%a.out
#SBATCH --error=logs/sobol_%A_%a.err

## Load modules
module load conda
conda activate rad_gen


## Prevent libraries from spawning their own threads on top of your workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs data/batches

srun python generate_sobol_samples.py \
    --batch-id   $SLURM_ARRAY_TASK_ID \
    --n-points   2097152 \
    --n-workers  $SLURM_CPUS_PER_TASK \
    --output-dir data/batches