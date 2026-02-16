# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import os
import sys
import pickle
import argparse

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import QED, Descriptors, Crippen

import dockstring
from dockstring import load_target
import multiprocessing as mp

from sklearn.preprocessing import MinMaxScaler

import math
import warnings

warnings.filterwarnings('ignore')

# %%
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# %%
parser = argparse.ArgumentParser(description='Analyze generated molecules')
parser.add_argument('--checkpoint_dir', type=str, required=True, 
                    help='Directory containing generated_results.pkl')
parser.add_argument('--properties', nargs='+', required=True, 
                    help='Properties used (e.g., --properties affinity logps qeds sas tpsas)')
args = parser.parse_args()

# %%
# Load the original dataset to fit scalers
df = pd.read_csv('../data/lck_dockstring_data1.csv')

affinity_scaler = MinMaxScaler()
qed_scaler = MinMaxScaler()
logp_scaler = MinMaxScaler()
tpsas_scaler = MinMaxScaler()
sas_scaler = MinMaxScaler()

affinity_scaler.fit(df['affinity'].values.reshape(-1,1))
qed_scaler.fit(df['qed'].values.reshape(-1,1))
logp_scaler.fit(df['logp'].values.reshape(-1,1))
tpsas_scaler.fit(df['tpsa'].values.reshape(-1,1))
sas_scaler.fit(df['sas'].values.reshape(-1,1))

# %%
def calc_properties(properties, smiles):
    """Calculate molecular properties for a list of SMILES."""
    qeds = []
    logps = []
    tpsas = []
    sas = []
    molwt = []
    props = []
    valid_smi = []
    
    for prop, smi in zip(properties, smiles):
        mol = Chem.MolFromSmiles(smi)
        try:
            if mol is not None:
                qed = QED.qed(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                sa = sascorer.calculateScore(mol)
                mw = Descriptors.MolWt(mol)
                
                qeds.append(qed)
                logps.append(logp)
                tpsas.append(tpsa)
                sas.append(sa)
                molwt.append(mw)
                props.append(prop)    
                valid_smi.append(smi)            
        except:
            pass
                
    return qeds, logps, tpsas, sas, molwt, np.array(props), valid_smi

# %%
def dock_smile(smile):
    """Calculate binding affinity for a single SMILES using dockstring."""
    target = load_target('LCK')
    try:
        score, _ = target.dock(smile)
    except:
        score = None
    return score

def calculate_binding_affinities(smiles_list, batch_size=16):
    """Calculate binding affinities for a list of SMILES using multiprocessing."""
    print(f"Calculating binding affinities for {len(smiles_list)} molecules...")
    
    # Pad to multiple of batch_size
    smiles_array = np.array(smiles_list)
    if len(smiles_array) % batch_size != 0:
        padding = batch_size - (len(smiles_array) % batch_size)
        smiles_array = np.concatenate([smiles_array, smiles_array[:padding]])
    
    smiles_batches = smiles_array.reshape(batch_size, -1)
    
    all_scores = []
    for batch in tqdm(smiles_batches, desc="Docking batches"):
        with mp.Pool(mp.cpu_count()) as pool:
            scores = pool.map(dock_smile, batch)
        all_scores.extend(scores)
    
    # Return only the original length (remove padding)
    all_scores = all_scores[:len(smiles_list)]
    return np.array(all_scores)

# %%
def load_and_calculate_properties(checkpoint_dir):
    """Load generated molecules and calculate their properties."""
    
    # Load generated results
    results_path = os.path.join(checkpoint_dir, 'generated_molecules.pkl')
    with open(results_path, 'rb') as f:
        results_dict = pickle.load(f)
    
    print(f"Loaded {len(results_dict)} query results")
    
    # Calculate properties for each query
    final_results = {}
    
    for key, samples in tqdm(results_dict.items(), desc="Calculating properties"):
        print(f"\nProcessing {key}...")
        
        # Calculate chemical properties
        qeds, logps, tpsas, sas, molwt, og_props, smi = calc_properties([-1]*len(samples), samples)
        
        # Calculate binding affinities using dockstring
        affinities = calculate_binding_affinities(smi)
        
        final_results[key] = [logps, qeds, sas, tpsas, affinities, samples]
    
    # Save final results with all properties
    output_path = os.path.join(checkpoint_dir, 'generated_results_with_properties.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"\nSaved results to: {output_path}")
    
    return final_results

# %%
def create_dataframe(results_dict, prop_names):
    """Create a DataFrame from results dictionary."""
    data = []
    columns = []
    
    for prop in prop_names:
        columns.append(f'Target {prop}')
    for prop in ['logps', 'qeds', 'sas', 'tpsas', 'affinity']:
        columns.append(f'Predicted {prop}')
    columns.append('key')
    
    for key in results_dict:
        # keys were created as strings (e.g. '3.5_0.7_...').
        # Try to convert each key-part to float so Target columns are numeric.
        raw_keys = key.split('_')
        conv_keys = []
        for k in raw_keys:
            try:
                conv_keys.append(float(k))
            except Exception:
                conv_keys.append(k)

        for i in range(len(results_dict[key][0])):
            # Filter out None affinities
            if results_dict[key][4][i] is not None:
                row = conv_keys + [
                    results_dict[key][0][i],  # logps
                    results_dict[key][1][i],  # qeds
                    results_dict[key][2][i],  # sas
                    results_dict[key][3][i],  # tpsas
                    results_dict[key][4][i],  # affinity
                    key
                ]
                data.append(row)
    
    data_df = pd.DataFrame(data, columns=columns)

    # Ensure numeric columns are numeric (coerce non-convertible values to NaN).
    for col in columns:
        if col.startswith('Target') or col.startswith('Predicted'):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    return data_df

def _gaussian_pdf(x, mu, sigma):
    coef = 1.0 / (math.sqrt(2 * math.pi) * sigma)
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coef * np.exp(exponent)

def compute_metrics_for_property(data_df, prop_name, checkpoint_dir, sigma=0.05):
    """Compute MAE, variance, and KL divergence (predicted vs Gaussian(target,sigma))
    for each unique target property value.

    Groups rows by unique target values and computes metrics within each group.
    Saves results to metrics_<prop_name>.csv in checkpoint_dir and returns the DataFrame.
    """
    target_col = f'Target {prop_name}'
    pred_col = f'Predicted {prop_name}'

    if target_col not in data_df.columns or pred_col not in data_df.columns:
        print(f"Skipping {prop_name}: columns not found")
        return None

    df = data_df[[target_col, pred_col]].dropna()
    if df.shape[0] == 0:
        print(f"No data for {prop_name}")
        return None

    # Get unique target values and sort them
    unique_targets = sorted(df[target_col].unique())
    results = []
    eps = 1e-12

    for target_val in unique_targets:
        # Get all rows with this target value
        mask = df[target_col] == target_val
        tvals = df.loc[mask, target_col].values
        preds = df.loc[mask, pred_col].values

        if len(preds) == 0:
            results.append({
                'target_value': target_val,
                'count': 0,
                'mae': np.nan,
                'var_pred': np.nan,
                'kl_div': np.nan
            })
            continue

        # MAE between predicted and target
        mae = np.mean(np.abs(preds - tvals))
        
        # Variance of predictions
        var_pred = np.var(preds)

        # KL divergence between empirical predicted distribution and Gaussian(target_val, sigma)
        # empirical p from histogram
        counts, edges = np.histogram(preds, bins=20)
        p_counts = counts.astype(float)
        if p_counts.sum() == 0:
            kl = np.nan
        else:
            p_prob = p_counts / p_counts.sum()
            centers = 0.5 * (edges[:-1] + edges[1:])
            # Gaussian centered at the target value with std=sigma
            q_vals = _gaussian_pdf(centers, target_val, sigma)
            q_prob = q_vals / (q_vals.sum() + eps)
            p_safe = p_prob + eps
            q_safe = q_prob + eps
            kl = np.sum(p_safe * np.log(p_safe / q_safe))

        results.append({
            'target_value': target_val,
            'count': int(len(preds)),
            'mae': float(mae),
            'var_pred': float(var_pred),
            'kl_div': float(kl)
        })

    res_df = pd.DataFrame(results)
    out_path = os.path.join(checkpoint_dir, f'metrics_{prop_name}.csv')
    res_df.to_csv(out_path, index=False)
    print(f"Saved metrics for {prop_name} to: {out_path}")
    return res_df

def compute_all_metrics(data_df, prop_names, checkpoint_dir):
    # compute metrics for each property available in the dataframe
    for prop in ['logps','qeds','sas','tpsas','affinity']:
        compute_metrics_for_property(data_df, prop, checkpoint_dir)

# %%
def plot_single_property_kdes(data_df, prop_names, target_props, results_dir):
    """Generate single property KDE plots."""
    print("\nGenerating single property KDE plots...")
    
    if 'logps' in target_props or 'logp' in target_props:
        plt.figure(figsize=(10, 6))
        target_col = f"Target {prop_names[prop_names.index('logps') if 'logps' in prop_names else prop_names.index('logp')]}"
        ax = sns.kdeplot(data=data_df, x="Predicted logps", hue=target_col, 
                        fill=True, alpha=0.5, linewidth=1, bw_adjust=2)
        plt.title('LogP Distribution (Generated vs Target)')
        plt.savefig(os.path.join(results_dir, 'logp_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: logp_kde.png")
    
    if 'qeds' in target_props or 'qed' in target_props:
        plt.figure(figsize=(10, 6))
        target_col = f"Target {prop_names[prop_names.index('qeds') if 'qeds' in prop_names else prop_names.index('qed')]}"
        ax = sns.kdeplot(data=data_df, x="Predicted qeds", hue=target_col, 
                        fill=True, alpha=0.5, linewidth=1, bw_adjust=2)
        plt.title('QED Distribution (Generated vs Target)')
        plt.savefig(os.path.join(results_dir, 'qed_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: qed_kde.png")
    
    if 'tpsas' in target_props or 'tpsa' in target_props:
        plt.figure(figsize=(10, 6))
        target_col = f"Target {prop_names[prop_names.index('tpsas') if 'tpsas' in prop_names else prop_names.index('tpsa')]}"
        ax = sns.kdeplot(data=data_df, x="Predicted tpsas", hue=target_col, 
                        fill=True, alpha=0.5, linewidth=1, bw_adjust=2)
        plt.title('TPSA Distribution (Generated vs Target)')
        plt.savefig(os.path.join(results_dir, 'tpsa_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: tpsa_kde.png")
    
    if 'sas' in target_props:
        plt.figure(figsize=(10, 6))
        target_col = f"Target {prop_names[prop_names.index('sas')]}"
        ax = sns.kdeplot(data=data_df, x="Predicted sas", hue=target_col, 
                        fill=True, alpha=0.5, linewidth=1, bw_adjust=2)
        plt.title('SAS Distribution (Generated vs Target)')
        plt.savefig(os.path.join(results_dir, 'sas_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: sas_kde.png")
    
    if 'affinity' in target_props:
        plt.figure(figsize=(10, 6))
        target_col = f"Target {prop_names[prop_names.index('affinity')]}"
        ax = sns.kdeplot(data=data_df, x="Predicted affinity", hue=target_col, 
                        fill=True, alpha=0.5, linewidth=1, bw_adjust=2)
        plt.title('Binding Affinity Distribution (Generated vs Target)')
        plt.xlabel('Binding Affinity (kcal/mol)')
        plt.savefig(os.path.join(results_dir, 'affinity_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: affinity_kde.png")

# %%
def plot_dual_property_kdes(data_df, prop_names, target_props, results_dir):
    """Generate dual property 2D KDE plots."""
    print("\nGenerating dual property KDE plots...")
    
    # TPSA vs LogP
    if ('tpsas' in target_props or 'tpsa' in target_props) and ('logps' in target_props or 'logp' in target_props):
        tpsa_col = f"Target {prop_names[prop_names.index('tpsas') if 'tpsas' in prop_names else prop_names.index('tpsa')]}"
        logp_col = f"Target {prop_names[prop_names.index('logps') if 'logps' in prop_names else prop_names.index('logp')]}"
        data_df['TPSA-LOGP'] = data_df[tpsa_col].astype(str) + '-' + data_df[logp_col].astype(str)
        plt.figure(figsize=(12, 8))
        ax = sns.kdeplot(data=data_df, x="Predicted tpsas", y="Predicted logps", hue="TPSA-LOGP", 
                        fill=True, alpha=0.7, bw_adjust=1.5)
        plt.title('TPSA vs LogP (Dual Property Conditioning)')
        plt.savefig(os.path.join(results_dir, 'tpsa_logp_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: tpsa_logp_kde.png")
    
    # TPSA vs SAS
    if ('tpsas' in target_props or 'tpsa' in target_props) and 'sas' in target_props:
        tpsa_col = f"Target {prop_names[prop_names.index('tpsas') if 'tpsas' in prop_names else prop_names.index('tpsa')]}"
        sas_col = f"Target {prop_names[prop_names.index('sas')]}"
        data_df['TPSA-SAS'] = data_df[tpsa_col].astype(str) + '-' + data_df[sas_col].astype(str)
        plt.figure(figsize=(12, 8))
        ax = sns.kdeplot(data=data_df, x="Predicted tpsas", y="Predicted sas", hue="TPSA-SAS", 
                        fill=True, alpha=0.7, bw_adjust=1.5)
        plt.title('TPSA vs SAS (Dual Property Conditioning)')
        plt.savefig(os.path.join(results_dir, 'tpsa_sas_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: tpsa_sas_kde.png")
    
    # QED vs LogP
    if ('qeds' in target_props or 'qed' in target_props) and ('logps' in target_props or 'logp' in target_props):
        qed_col = f"Target {prop_names[prop_names.index('qeds') if 'qeds' in prop_names else prop_names.index('qed')]}"
        logp_col = f"Target {prop_names[prop_names.index('logps') if 'logps' in prop_names else prop_names.index('logp')]}"
        data_df['QED-LOGP'] = data_df[qed_col].astype(str) + '-' + data_df[logp_col].astype(str)
        plt.figure(figsize=(12, 8))
        ax = sns.kdeplot(data=data_df, x="Predicted qeds", y="Predicted logps", hue="QED-LOGP", 
                        fill=True, alpha=0.7, bw_adjust=1.5)
        plt.title('QED vs LogP (Dual Property Conditioning)')
        plt.savefig(os.path.join(results_dir, 'qed_logp_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: qed_logp_kde.png")
    
    # SAS vs LogP
    if 'sas' in target_props and ('logps' in target_props or 'logp' in target_props):
        sas_col = f"Target {prop_names[prop_names.index('sas')]}"
        logp_col = f"Target {prop_names[prop_names.index('logps') if 'logps' in prop_names else prop_names.index('logp')]}"
        data_df['SAS-LOGP'] = data_df[sas_col].astype(str) + '-' + data_df[logp_col].astype(str)
        plt.figure(figsize=(12, 8))
        ax = sns.kdeplot(data=data_df, x="Predicted sas", y="Predicted logps", hue="SAS-LOGP", 
                        fill=True, alpha=0.7, bw_adjust=1.5)
        plt.title('SAS vs LogP (Dual Property Conditioning)')
        plt.savefig(os.path.join(results_dir, 'sas_logp_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: sas_logp_kde.png")
    
    # Affinity vs LogP
    if 'affinity' in target_props and ('logps' in target_props or 'logp' in target_props):
        aff_col = f"Target {prop_names[prop_names.index('affinity')]}"
        logp_col = f"Target {prop_names[prop_names.index('logps') if 'logps' in prop_names else prop_names.index('logp')]}"
        data_df['AFFINITY-LOGP'] = data_df[aff_col].astype(str) + '-' + data_df[logp_col].astype(str)
        plt.figure(figsize=(12, 8))
        ax = sns.kdeplot(data=data_df, x="Predicted affinity", y="Predicted logps", hue="AFFINITY-LOGP", 
                        fill=True, alpha=0.7, bw_adjust=1.5)
        plt.title('Binding Affinity vs LogP (Dual Property Conditioning)')
        plt.xlabel('Binding Affinity (kcal/mol)')
        plt.ylabel('LogP')
        plt.savefig(os.path.join(results_dir, 'affinity_logp_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: affinity_logp_kde.png")
    
    # Affinity vs TPSA
    if 'affinity' in target_props and ('tpsas' in target_props or 'tpsa' in target_props):
        aff_col = f"Target {prop_names[prop_names.index('affinity')]}"
        tpsa_col = f"Target {prop_names[prop_names.index('tpsas') if 'tpsas' in prop_names else prop_names.index('tpsa')]}"
        data_df['AFFINITY-TPSA'] = data_df[aff_col].astype(str) + '-' + data_df[tpsa_col].astype(str)
        plt.figure(figsize=(12, 8))
        ax = sns.kdeplot(data=data_df, x="Predicted affinity", y="Predicted tpsas", hue="AFFINITY-TPSA", 
                        fill=True, alpha=0.7, bw_adjust=1.5)
        plt.title('Binding Affinity vs TPSA (Dual Property Conditioning)')
        plt.xlabel('Binding Affinity (kcal/mol)')
        plt.ylabel('TPSA')
        plt.savefig(os.path.join(results_dir, 'affinity_tpsa_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: affinity_tpsa_kde.png")
    
    # Affinity vs QED
    if 'affinity' in target_props and ('qeds' in target_props or 'qed' in target_props):
        aff_col = f"Target {prop_names[prop_names.index('affinity')]}"
        qed_col = f"Target {prop_names[prop_names.index('qeds') if 'qeds' in prop_names else prop_names.index('qed')]}"
        data_df['AFFINITY-QED'] = data_df[aff_col].astype(str) + '-' + data_df[qed_col].astype(str)
        plt.figure(figsize=(12, 8))
        ax = sns.kdeplot(data=data_df, x="Predicted affinity", y="Predicted qeds", hue="AFFINITY-QED", 
                        fill=True, alpha=0.7, bw_adjust=1.5)
        plt.title('Binding Affinity vs QED (Dual Property Conditioning)')
        plt.xlabel('Binding Affinity (kcal/mol)')
        plt.ylabel('QED')
        plt.savefig(os.path.join(results_dir, 'affinity_qed_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: affinity_qed_kde.png")
    
    # Affinity vs SAS
    if 'affinity' in target_props and 'sas' in target_props:
        aff_col = f"Target {prop_names[prop_names.index('affinity')]}"
        sas_col = f"Target {prop_names[prop_names.index('sas')]}"
        data_df['AFFINITY-SAS'] = data_df[aff_col].astype(str) + '-' + data_df[sas_col].astype(str)
        plt.figure(figsize=(12, 8))
        ax = sns.kdeplot(data=data_df, x="Predicted affinity", y="Predicted sas", hue="AFFINITY-SAS", 
                        fill=True, alpha=0.7, bw_adjust=1.5)
        plt.title('Binding Affinity vs SAS (Dual Property Conditioning)')
        plt.xlabel('Binding Affinity (kcal/mol)')
        plt.ylabel('SAS')
        plt.savefig(os.path.join(results_dir, 'affinity_sas_kde.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: affinity_sas_kde.png")

# %%
def main():
    checkpoint_dir = args.checkpoint_dir
    prop_names = args.properties
    
    print("="*50)
    print(f"Analyzing molecules from: {checkpoint_dir}")
    print(f"Properties: {prop_names}")
    print("="*50)
    
    # Load and calculate properties
    results_dict = load_and_calculate_properties(checkpoint_dir)
    
    # Determine which properties were used as targets
    target_props = {}
    if 'affinity' in prop_names:
        target_props['affinity'] = True
    if 'logps' in prop_names or 'logp' in prop_names:
        target_props['logps'] = True
    if 'qeds' in prop_names or 'qed' in prop_names:
        target_props['qeds'] = True
    if 'tpsas' in prop_names or 'tpsa' in prop_names:
        target_props['tpsas'] = True
    if 'sas' in prop_names:
        target_props['sas'] = True
    
    # Create DataFrame
    data_df = create_dataframe(results_dict, prop_names)
    data_df.to_csv(os.path.join(checkpoint_dir, 'generated_data.csv'), index=False)
    print(f"\nSaved DataFrame to: {os.path.join(checkpoint_dir, 'generated_data.csv')}")
    # Compute MAE, variance, and KL divergence per-property across target-value bins
    # Load previously saved results with properties
    data_df = pd.read_csv(os.path.join(checkpoint_dir, 'generated_data.csv'))
    print("\nComputing per-property metrics (MAE, variance, KL vs Gaussian sigma=0.05)...")
    compute_all_metrics(data_df, prop_names, checkpoint_dir)
    
    # Plot single property KDEs
    plot_single_property_kdes(data_df, prop_names, target_props, checkpoint_dir)
    
    # Plot dual property KDEs
    plot_dual_property_kdes(data_df, prop_names, target_props, checkpoint_dir)
    
    print("\n" + "="*50)
    print(f"All results saved to: {checkpoint_dir}")
    print("="*50 + "\n")

# %%
if __name__ == "__main__":
    main()
