#!/bin/bash
#SBATCH --job-name=PREC_SHEAR_TRACKING
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_output/track-%A_%a.out
#SBATCH --array=0
#SBATCH --exclude=node092,node113,node029,node034

# ==========================
# Prepare environment
# ==========================
mkdir -p slurm_output

module use $HOME/modulefiles
module load atom_sims

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ==========================
# Paths & input files
# ==========================
SCRIPT_PATH="/mnt/parscratch/users/mtp24ele/private/prec_interactions/dislo_tracking/01_shear/run.py"

INPUT_FILE_DIR="/mnt/parscratch/users/mtp24ele/private/prec_interactions/data/shear"
INPUT_FILES=("shear_T800_SR1e8_R20_N1000" "shear_T800_SR1e7_R20_N1000"  "shear_T800_SR1e7_R30_N1000"  "shear_T800_SR1e7_R40_N1000")

# --------------------------
# Select array-specific input
# --------------------------
INPUT_FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}

# ==========================
# Run simulation
# ==========================
srun --export=ALL python "${SCRIPT_PATH}" --input "${INPUT_FILE_DIR}/${INPUT_FILE}"