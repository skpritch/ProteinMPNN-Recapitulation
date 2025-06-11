#!/usr/bin/env bash
#SBATCH --job-name=mpnn_vanilla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12              # 12 CPUs/task
#SBATCH --mem=128G
#SBATCH --time=72:00:00                # HH:MM:SS
#SBATCH --partition=comino
#SBATCH --gres=gpu:1
#SBATCH --output=logs/vanilla_%j.out

# --- Global hyperparameters -----------------------
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-10000}
NEIGHBORS=${NEIGHBORS:-48}
NOISE=${NOISE:-0.02}

# Create a unique job tag based on hyperparameters
JOB_TAG="e${EPOCHS}_b${BATCH_SIZE}_n${NEIGHBORS}_noise${NOISE}"

# Define output and log directories
OUTPUT_DIR="outputs/vanilla_${JOB_TAG}"
LOG_DIR="logs"

# --- Verify and create directories ----------------
if [ -d "${OUTPUT_DIR}" ]; then
  echo "Output directory ${OUTPUT_DIR} already exists, continuing."
else
  mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
fi

# --- Environment setup -----------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpnn_env
echo "Active conda environment: $(conda info --envs)"

# --- Launch training -------------------------------
echo "Starting ProteinMPNN vanilla training:"
echo "  epochs       = $EPOCHS"
echo "  batch size   = $BATCH_SIZE"
echo "  neighbors    = $NEIGHBORS"
echo "  noise level  = $NOISE"
echo "  output dir   = $OUTPUT_DIR"

python training/training.py \
  --path_for_training_data training/data/pdb_2021aug02 \
  --path_for_outputs "${OUTPUT_DIR}" \
  --num_epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_neighbors "${NEIGHBORS}" \
  --backbone_noise "${NOISE}" \
  --mixed_precision True

echo "Job completed successfully."