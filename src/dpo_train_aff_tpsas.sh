#!/bin/bash

#SBATCH --job-name=dpo_aff_tpsa_generate_molecules
#SBATCH -A research
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=dpo_train_affinity_tpsa.out
#SBATCH --error=dpo_train_affinity_tpsa.err
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 dpo_training_generation.py --properties affinity tpsas
