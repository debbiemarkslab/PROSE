#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu_quad
#SBATCH -t 2-00:00
#SBATCH --mem=100G
#SBATCH --output=/n/groups/marks/users/erik/Promoter_Poet_private/slurm/sei%j.out
#SBATCH --error=/n/groups/marks/users/erik/Promoter_Poet_private/slurm/sei%j.err
#SBATCH --job-name="prose_sei"
#SBATCH --array=0

source ~/.bashrc

# Load necessary modules
module load gcc/9.2.0
module load cuda/12.1

conda activate /n/groups/marks/software/anaconda_o2/envs/sei

seqs='/n/groups/marks/users/erik/Promoter_Poet_private/data/rand.fasta'
# Define output files
output_run=/n/groups/marks/users/erik/Promoter_Poet_private/sei
predict_path="$output_run/chromatin-profiles-hdf5/rand_predictions.h5"

# Create the output directory structure
mkdir -p "$output_run"

# Change directory
cd /n/groups/marks/users/courtney/projects/regulatory_genomics/models/prior_art/Sei/sei-framework

# Run prediction and scoring scripts
bash 1_sequence_prediction.sh "$seqs" hg38 "$output_run" --cuda
sh 2_raw_sc_score.sh "$predict_path" "$output_run"

