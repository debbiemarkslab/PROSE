# Promoter_Poet_private
local copy at `/n/groups/marks/users/erik/Promoter_Poet_private`
## Environment
use `conda activate promoet`

full env path `/home/jix836/.conda/envs/promoet`
## Training
without human token: `python scripts/train.py`

with human token: `python scripts/train_token.py`

slurm script that works with both: `train_o2.sh`

relevant args: `--max_len` (total length of each set, default = 16k), `--batch_size` (change for CUDA OOM errors, default = 2)
## Generation
without token: `scripts/generate.py`

with token: `scripts/generate_token.py`

slurm script that works with both: `score.sh`

relevant args: `--output_csv_path`, `--batch_size` (# seqs generated per prompt), `--ckpt_path`, `--temp`
## Evaluation
`scripts\notebook.ipynb`
