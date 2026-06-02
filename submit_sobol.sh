#!/bin/bash
#SBATCH --job-name=rad_sobol
#SBATCH --array=0-1              # 100 independent batches → adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # Must match --n-workers below
#SBATCH --mem=16G
#SBATCH --time=04:00:00
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

srun python generate_sobol_batch.py \
    --batch-id   $SLURM_ARRAY_TASK_ID \
    --n-points   4096 \
    --n-workers  8 \
    --output-dir data/batches