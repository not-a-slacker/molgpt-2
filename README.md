# MolGPT-2

MolGPT-2 (MolGPT2.0) — Multi-Objective Molecule Generation using a Transformer
encoder-decoder and Direct Preference Optimization (DPO).

This repository implements the work described in:

MolGPT2.0: Multi-Objective Molecule Generation using GPT Architecture and Preference Optimization

Abstract: MolGPT2.0 conditions molecular generation on binding affinity and other
physicochemical properties using a transformer encoder-decoder architecture and
uses Direct Preference Optimization (DPO) to further refine generation toward
preferred molecules. The model produces valid, novel, and target-specific SMILES
that optimize binding affinity (docking score) while maintaining desirable
chemical properties.

-- Project layout

- `src/` : main scripts and notebooks for training, generation, analysis and plotting.
  - [src/bindingAffinity_dockstring_multiproperty_decoder_arc.py](src/bindingAffinity_dockstring_multiproperty_decoder_arc.py) : encoder-decoder training & sampling.
  - [src/dpo_training_generation.py](src/dpo_training_generation.py) : DPO fine-tuning and generation pipeline.
  - [src/analyze_generated_molecules.py](src/analyze_generated_molecules.py) : compute RDKit properties, docking and metrics for generated molecules.
  - [src/plot_comparison.py](src/plot_comparison.py) : comparison KDE plots between checkpoints.
  - [src/CreatePreferenceDataFullProperties.py](src/CreatePreferenceDataFullProperties.py) : build preference datasets.
  - [src/CreatePreferenceDataAffinityProperty.py](src/CreatePreferenceDataAffinityProperty.py) : preference dataset focused on affinity.
  - Notebooks: [src/compute_dockstring_properties.ipynb](src/compute_dockstring_properties.ipynb), etc.

- `data\` : Contains dataset used for training model (DOCKSTRING dataset,consists of around 260k molecules and their binding affinities with ~50 protein targets).

-- Requirements & installation

Recommended: create a conda environment to install RDKit and other scientific packages.

1. Create environment and install Dockstring and RDKit(conda-forge):

```bash
conda create -n molgpt2 python=3.9 -y
conda activate molgpt2
conda install -c conda-forge rdkit=20239.* -y
conda install conda-forge::dockstring
```

2. Install remaining Python packages (pip):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA version or use cpu wheels
pip install pandas numpy scikit-learn tqdm matplotlib seaborn sascorer
```


-- Data

- The primary dataset used in the code is `data/lck_dockstring_data1.csv` (DOCKSTRING-derived).
- Generated outputs and checkpoints are stored under `../checkpoints/<run_name>/` by the training scripts.

-- Examples

1. Train / run the encoder-decoder model (example):

```bash
python src/bindingAffinity_dockstring_multiproperty_decoder_arc.py --properties affinity logps qeds sas tpsas
```

2. Create preference data (full properties):

```bash
python src/CreatePreferenceDataFullProperties.py
```

3. Run DPO fine-tuning 

```bash
python src/dpo_training_generation.py --model_properties affinity logps qeds sas tpsas --preference_properties affinity
```

4. Analyze generated molecules and compute property metrics:

```bash
python src/analyze_generated_molecules.py --checkpoint_dir ../checkpoints/<run_name> --properties affinity logps qeds sas tpsas
```

5. Compare two checkpoints (KDE plots):

```bash
python src/plot_comparison.py --checkpoint_dir1 ../checkpoints/runA --checkpoint_dir2 ../checkpoints/runB --properties affinity logps qeds sas tpsas
```

-- Expected outputs

- `../checkpoints/<run_name>/generated_molecules.pkl` — raw generated samples per query.
- `../checkpoints/<run_name>/generated_results_with_properties.pkl` — generated samples with computed RDKit properties and docking scores.
- `../checkpoints/<run_name>/generated_data.csv` — flattened CSV for plotting and metrics.
- `../checkpoints/<run_name>/metrics_<property>.csv` — per-target MAE/KL metrics produced by `analyze_generated_molecules.py`.

-- Figures & metrics

- Use `analyze_generated_molecules.py` to calculate properties (RDKit + dockstring) and generate `generated_data.csv`.
- Use `plot_comparison.py` and helper plotting scripts in `src/` to reproduce KDEs, 2D KDEs, and target-specific comparisons.

-- Notes on training and DPO

- The encoder network conditions on concatenated, normalized property vectors (min-max scaled from training data).
- The decoder generates SMILES autoregressively with a masked attention mechanism.
- DPO fine-tunes the model using preference pairs (preferred vs unpreferred samples) to nudge generation toward higher affinity and other property alignment.
