#!/bin/bash

#SBATCH --job-name=affinity_tpsa_training_dualproperty_46_epochs
#SBATCH -A research
#SBATCH -c 18
#SBATCH --mem-per-cpu=2G
#SBATCH -w gnode084
#SBATCH --gres=gpu:2
#SBATCH --output=affinity_tpsa_train.txt
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 affinity_tpsa_dockstring_dualproperty_encoder_decoder.py
