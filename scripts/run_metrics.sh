#!/bin/bash
#SBATCH --job-name=submodular_nn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

mkdir -p logs

module load anaconda3/2023.03
module load cuda/12.6.1

# source $(conda info --base)/etc/profile.d/conda.sh
conda activate sub_nn

cd $SLURM_SUBMIT_DIR

wandb login $WANDB_API_KEY

python3 metrics.py
