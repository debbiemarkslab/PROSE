#!/bin/bash 
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -p gpu_quad
#SBATCH -t 1-00:00
#SBATCH --mem=64GB
#SBATCH --output=/n/groups/marks/users/erik/Promoter_Poet_private/slurm/prose%j.out
#SBATCH --error=/n/groups/marks/users/erik/Promoter_Poet_private/slurm/prose%j.err
#SBATCH --job-name="prose"

source activate base
conda activate promoet
export LD_LIBRARY_PATH=/home/jix836/.conda/envs/promoet/lib:${LD_LIBRARY_PATH}
cd /n/groups/marks/users/erik/Promoter_Poet_private
python scripts/tokenizer.py
