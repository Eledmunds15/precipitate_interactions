#!/bin/bash
#SBATCH --job-name=PREC_SHEAR
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=30:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_shear/slurm-%A_%a.out
#SBATCH --array=0-2
#SBATCH --exclude=node092,node113,node029,node034

mkdir -p slurm_shear

# 1. Load your modules
module use $HOME/modulefiles
module load atom_sims

# 2. Define the inputs
INPUT_DIR="/mnt/parscratch/users/mtp24ele/private/prec_interactions/input"
INPUT_FILES=(Fe_E111_110_R20.lmp Fe_E111_110_R30.lmp Fe_E111_110_R40.lmp)

# Use the Task ID to select the specific file (0, 1, or 2)
FILE_TO_RUN=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}

# 3. Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4. Run the simulation
srun --export=ALL python simulations/01_shear.py \
    --temperature 800 \
    --strain_rate 1e7 \
    --input "${INPUT_DIR}/${FILE_TO_RUN}" \
    --run_time 1500000