#!/bin/bash

#SBATCH --job-name=logp_training_singleproperty_100_epochs
#SBATCH -A research
#SBATCH -c 40
#SBATCH -w gnode091
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:4
#SBATCH --output=logp_train.txt
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 logp_dockstring_singleproperty_encoder_decoder.py
