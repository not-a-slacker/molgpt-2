#!/bin/bash

#SBATCH --job-name=qed_training_singleproperty_1000_epochs
#SBATCH -A research
#SBATCH -w gnode069
#SBATCH -c 36
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:4
#SBATCH --output=qed_train.txt
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 qed_dockstring_singleproperty_encoder_decoder.py
