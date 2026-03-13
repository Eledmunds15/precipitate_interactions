#!/bin/bash
#SBATCH --job-name=PREC_SHEAR_TRACKING
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_shear/slurm-tracking-%A_%a.out
#SBATCH --array=0-2
#SBATCH --exclude=node092,node113,node029,node034

mkdir -p slurm_shear

# 1. Load your modules
module use $HOME/modulefiles
module load atom_sims

# 2. Define the inputs
INPUT_DIR="/mnt/parscratch/users/mtp24ele/private/prec_interactions/data"
INPUT_FILES=("shear_T800_SR1E7_R20_N4283" "shear_T800_SR1E7_R30_N3350" "shear_T800_SR1E7_R40_N8562")

# Use the Task ID to select the specific file (0, 1, or 2)
FILE_TO_RUN=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}

# 3. Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4. Run the simulation
srun --export=ALL python dislo_tracking/01_track_shear.py \
    --input "${INPUT_DIR}/${FILE_TO_RUN}"