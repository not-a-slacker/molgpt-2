#!/bin/bash

#SBATCH --job-name=affinity_sas_training_dualproperty_125_epochs
#SBATCH -A research
#SBATCH -w gnode080

#SBATCH -c 18
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:2
#SBATCH --output=affinity_sas_train.txt
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 affinity_sas_dockstring_dualproperty_encoder_decoder.py
