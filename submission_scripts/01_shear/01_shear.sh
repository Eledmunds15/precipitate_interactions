#!/bin/bash
#SBATCH --job-name=PREC_SHEAR
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=30:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_output/run-%A_%a.out
#SBATCH --array=0-2
#SBATCH --exclude=node111,node089,node119

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
SCRIPT_PATH="/mnt/parscratch/users/mtp24ele/private/prec_interactions/simulations/01_shear/run.py"
POTENTIAL_PATH="/mnt/parscratch/users/mtp24ele/private/prec_interactions/potentials/mendelev03.fs"

INPUT_FILE_DIR="/mnt/parscratch/users/mtp24ele/private/prec_interactions/input"
INPUT_FILES=("Fe_E111_110_R20.lmp" "Fe_E111_110_R30.lmp" "Fe_E111_110_R40.lmp")
RADIUSES=(20 30 40)

# ==========================
# Simulation parameters
# ==========================
TEMPERATURE=800
STRAIN_RATE=1e7
RUNTIME=1500000
RANDOM_SEED=1000

# --------------------------
# Select array-specific input
# --------------------------
INPUT_FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}
RADIUS=${RADIUSES[$SLURM_ARRAY_TASK_ID]}

# --------------------------
# Simulation name
# --------------------------
NAME="shear_T${TEMPERATURE}_SR${STRAIN_RATE}_R${RADIUS}_N${RANDOM_SEED}"

# ==========================
# Run simulation
# ==========================
srun --export=ALL python "${SCRIPT_PATH}" \
    --temperature ${TEMPERATURE} \
    --strain_rate ${STRAIN_RATE} \
    --input "${INPUT_FILE_DIR}/${INPUT_FILE}" \
    --potential "${POTENTIAL_PATH}" \
    --run_time ${RUNTIME} \
    --random_seed ${RANDOM_SEED} \
    --name ${NAME}