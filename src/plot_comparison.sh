#!/bin/bash

#SBATCH --job-name=plot_comparison
#SBATCH -A research
#SBATCH -c 36
#SBATCH --mem-per-cpu=2G
#SBATCH --output=plot_comparison.out
#SBATCH --error=plot_comparison.err
#SBATCH --time=4-00:00:00

echo "Starting rn ..."
python3 plot_comparison.py --checkpoint_dir1 ../checkpoints/LCK_DOCKSTRING_FAST_ACTUAL_affinity_tpsas --label1 "All Properties (without DPO) Model" --checkpoint_dir2 ../checkpoints/DPO_Finetuning_BETA_0.11_epochs_10_LCK_DOCKSTRING_FAST_ACTUAL_affinity_tpsas_DPO_pref_affinity --label2 "All properties (with DPO) Model" --properties affinity tpsas
echo "Finished plotting comparison"