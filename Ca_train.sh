#!/usr/bin/env bash
#SBATCH --job-name=ca_only         # changed name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=comino
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ca_%j.out     # logs/ca_JOBID.out

# --- Global hyperparameters -----------------------
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-10000}
NEIGHBORS=${NEIGHBORS:-48}
NOISE=${NOISE:-0.00}

# Create a unique job tag based on hyperparameters
JOB_TAG="e${EPOCHS}_b${BATCH_SIZE}_n${NEIGHBORS}_noise${NOISE}"

# Define output and log directories
OUTPUT_DIR="outputs/ca_${JOB_TAG}"
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
echo "Environment: $(conda info --envs | grep '*' )"

# --- Launch training -------------------------------
echo "Starting ProteinMPNN Cα-only training:"
echo "  epochs     = $EPOCHS"
echo "  batch size = $BATCH_SIZE"
echo "  neighbors  = $NEIGHBORS"
echo "  noise      = $NOISE"
echo "  output dir = $OUTPUT_DIR"

python training/ca_training.py \
  --path_for_training_data training/data/pdb_2021aug02 \
  --path_for_outputs "${OUTPUT_DIR}" \
  --num_epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_neighbors "${NEIGHBORS}" \
  --backbone_noise "${NOISE}" \
  --mixed_precision True \
  --ca_only

echo "Cα-only training job completed."