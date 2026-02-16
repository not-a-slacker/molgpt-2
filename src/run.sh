#!/bin/bash

#SBATCH --job-name=plot_dpo_affinity
#SBATCH -A research
#SBATCH -c 36
#SBATCH --mem-per-cpu=2G
#SBATCH --output=dpo_plot_affinity.out
#SBATCH --error=dpo_plot_affinity.err
#SBATCH --time=4-00:00:00

echo "Starting ..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 analyze_generated_molecules.py --checkpoint_dir ../checkpoints/DPO_Finetuning_BETA_0.11_epochs_10_LCK_DOCKSTRING_FAST_ACTUAL_affinity_logps_DPO_pref_affinity --properties affinity logps
