#!/bin/bash

#SBATCH --job-name=sas_training_singleproperty_200_epochs
#SBATCH -A research
#SBATCH -w gnode049
#SBATCH -c 36
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:4
#SBATCH --output=sas_train.txt
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 sas_dockstring_singleproperty_encoder_decoder.py
