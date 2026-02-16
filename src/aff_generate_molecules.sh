#!/bin/bash

#SBATCH --job-name=orig_generate_molecules
#SBATCH -A research
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=orig_generate_mols.out
#SBATCH --error=orig_generate_mols.err
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 bindingAffinity_dockstring_multiproperty_decoder_arc.py --properties affinity logps
echo "Finished logps"