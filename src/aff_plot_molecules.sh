#!/bin/bash

#SBATCH --job-name=orig_plot_molecules
#SBATCH -A research
#SBATCH -c 36
#SBATCH --mem-per-cpu=2G
#SBATCH --output=orig_plot_mols.out
#SBATCH --error=orig_plot_mols.err
#SBATCH --time=4-00:00:00

echo "Starting ..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 analyze_generated_molecules.py --checkpoint_dir ../checkpoints/LCK_DOCKSTRING_FAST_ACTUAL_affinity_logps --properties affinity logps
echo "Finished logps"