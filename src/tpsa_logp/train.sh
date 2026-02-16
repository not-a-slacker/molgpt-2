#!/bin/bash

#SBATCH --job-name=tpsa_logp_training_dualproperty_400_epochs
#SBATCH -A research
#SBATCH -w gnode086
#SBATCH -c 9
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --output=tpsa_logp_train.txt
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 tpsa_logp_dockstring_dualproperty_encoder_decoder.py
