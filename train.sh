#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -p gpu_dia
#SBATCH -t 2-00:00
#SBATCH --mem=100G
#SBATCH --output=/n/lw_groups/hms/sysbio/marks/lab/jix836/Promoter_Poet_private/slurm/Promoet_dgx%j.out
#SBATCH --error=/n/lw_groups/hms/sysbio/marks/lab/jix836/Promoter_Poet_private/slurm/Promoet_dgx%j.err
#SBATCH --job-name="PromoET"

source ~/.bashrc

module load dgx
module load miniconda3
module load cuda/12.1

source activate base
conda activate promoet

cd /n/lw_groups/hms/sysbio/marks/lab/jix836/Promoter_Poet_private
python scripts/train.py