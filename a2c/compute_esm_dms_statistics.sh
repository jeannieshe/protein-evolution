#!/bin/bash
# SBATCH -p mit_normal_gpu
#SBATCH --job-name=stat
# SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=a2c/logs/%j-%x.out

# Load your shell environment to activate your Conda environment
source /home/jeanshe/.bashrc
conda activate plm
cd /home/jeanshe/orcd/pool/protein-evolution

python a2c/compute_esm_dms_statistics.py

echo "Command completed."