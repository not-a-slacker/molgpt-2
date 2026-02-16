#!/bin/bash

#SBATCH --job-name=all_generate_molecules
#SBATCH -A research
#SBATCH -c 39
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:4
#SBATCH --output=all_generate.out
#SBATCH --error=all_generate.error
#SBATCH --time=4-00:00:00
#SBATCH -w gnode055

echo "Starting ..."
python3 all_dockstring_multiproperty_encoder_decoder.py
