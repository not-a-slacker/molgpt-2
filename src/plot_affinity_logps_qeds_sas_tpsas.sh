#!/bin/bash

#SBATCH --job-name=property_all
#SBATCH -A research
#SBATCH -c 30
#SBATCH --mem-per-cpu=2G
#SBATCH --output=dpo_plot_affinity.out
#SBATCH --error=dpo_plot_affinity.err
#SBATCH --time=4-00:00:00
#SBATCH -w gnode058

echo "Starting ..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 analyze_generated_molecules.py --checkpoint_dir ../checkpoints/LCK_DOCKSTRING_FAST_ACTUAL_affinity_logps_qeds_sas_tpsas --properties affinity logps qeds sas tpsas
