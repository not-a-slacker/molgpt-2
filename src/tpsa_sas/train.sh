#!/bin/bash

#SBATCH --job-name=tpsa_sas_training_dualproperty_400_epochs
#SBATCH -A research
#SBATCH -c 9
#SBATCH -w gnode086
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --output=tpsa_sas_train.txt
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 tpsa_sas_dockstring_dualproperty_encoder_decoder.py
