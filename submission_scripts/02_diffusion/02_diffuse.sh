#!/bin/bash
#SBATCH --job-name=DIFF_RERUN
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=30:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_shear/diff-%A_%a.out
#SBATCH --array=0-7
#SBATCH --exclude=node111

mkdir -p slurm_shear

# 1. Load your modules
module use $HOME/modulefiles
module load atom_sims

# 2. Define the Base Directory on HPC
# This should point to the folder containing your R30_S150_T700 etc. folders
BASE_DIR="/mnt/parscratch/users/mtp24ele/private/prec_interactions/data/rerun"

# 3. Build an array of the case directories
# This looks for all directories in the BASE_DIR
CASE_DIRS=($(ls -d ${BASE_DIR}/*/))

# Use the Task ID to select the specific folder
# e.g., if TASK_ID=0, SELECTED_DIR might be .../R20_S150_T800/
SELECTED_DIR=${CASE_DIRS[$SLURM_ARRAY_TASK_ID]}

# 4. Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 5. Run the simulation
# Point --meta to the metadata.json inside the selected case folder
srun --export=ALL python simulations/02_diffusion.py \
    --meta "${SELECTED_DIR}/metadata.json" \
    --run_time 1500000