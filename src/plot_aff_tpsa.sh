#!/bin/bash

#SBATCH --job-name=tpsa_aff_calc_props
#SBATCH -A research
#SBATCH -c 30
#SBATCH --mem-per-cpu=2G
#SBATCH --output=plot_affinity_tpsa.out
#SBATCH --error=plot_affinity_tpsa.err
#SBATCH --time=4-00:00:00
#SBATCH -w gnode058

echo "Starting ..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 analyze_generated_molecules.py --checkpoint_dir ../checkpoints/LCK_DOCKSTRING_FAST_ACTUAL_affinity_tpsa --properties affinity tpsa
