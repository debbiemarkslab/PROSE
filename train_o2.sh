#!/bin/bash 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu_quad
#SBATCH -t 2-12:00
#SBATCH --mem=50GB
#SBATCH --output=/n/groups/marks/users/erik/Promoter_Poet_private/slurm/prose%j.out
#SBATCH --error=/n/groups/marks/users/erik/Promoter_Poet_private/slurm/prose%j.err
#SBATCH --job-name="prose"

source activate base
conda activate promoet
export LD_LIBRARY_PATH=/home/jix836/.conda/envs/promoet/lib:${LD_LIBRARY_PATH}
wandb login 6d15eff7893c02d9755493383bac182122241aae
cd /n/groups/marks/users/erik/Promoter_Poet_private
python scripts/train_token.py
