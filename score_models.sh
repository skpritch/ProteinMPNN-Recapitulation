#!/usr/bin/env bash
#SBATCH --job-name=mpnn_score      # job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12         # 12 CPUs/task (same as vanilla_train.sh)
#SBATCH --mem=128G                 # 128 GB RAM
#SBATCH --time=08:00:00            # 4 hours (adjust as needed)
#SBATCH --partition=comino         # same partition as in vanilla_train.sh
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --output=logs/score_%j.out # write logs to logs/score_<JOBID>.out

# --- Environment setup -----------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpnn_env
echo "Active conda environment: $(conda info --envs | grep '*' )"

# Define specific model input directory
MODEL_DIR="noise_spread"

# --- Run the scoring script ------------------------
python score_models.py \
    --model_folder   results/${MODEL_DIR}/ \
    --test_pdb_dir   testing/data/pdb_test/        \
    --out_csv        results/${MODEL_DIR}.csv      \
    --batch_size     16                               \
    --max_length     512                              \

echo "MPNN scoring job completed."
