#!/bin/bash

#SBATCH --job-name=tpsa_aff_calc_props
#SBATCH -A research
#SBATCH -c 36
#SBATCH --mem-per-cpu=2G
#SBATCH --output=dpo_plot_affinity_tpsa.out
#SBATCH --error=dpo_plot_affinity_tpsa.err
#SBATCH --time=4-00:00:00

echo "Starting ..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 analyze_generated_molecules.py --checkpoint_dir ../checkpoints/DPO_Finetuning_BETA_0.11_epochs_10_LCK_DOCKSTRING_FAST_ACTUAL_affinity_tpsas --properties affinity tpsas
