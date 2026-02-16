#!/bin/bash

#SBATCH --job-name=tpsa_training_singleproperty_200_epochs
#SBATCH -A research
#SBATCH -c 36
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:4
#SBATCH --output=tpsa_train.txt
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 tpsa_dockstring_singleproperty_encoder_decoder.py
