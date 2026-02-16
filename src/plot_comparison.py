import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Compare generated molecules from two checkpoints')
parser.add_argument('--checkpoint_dir1', type=str, required=True, 
                    help='First checkpoint directory containing generated_data.csv')
parser.add_argument('--checkpoint_dir2', type=str, required=True, 
                    help='Second checkpoint directory containing generated_data.csv')
parser.add_argument('--properties', nargs='+', required=True, 
                    help='Properties to compare (e.g., --properties logps qeds sas tpsas affinity)')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory for comparison plots (default: checkpoint_dir1/comparison)')
parser.add_argument('--label1', type=str, default='Checkpoint 1',
                    help='Label for first checkpoint in plots')
parser.add_argument('--label2', type=str, default='Checkpoint 2',
                    help='Label for second checkpoint in plots')
args = parser.parse_args()


def load_checkpoint_data(checkpoint_dir):
    """Load generated_data.csv from checkpoint directory."""
    data_path = os.path.join(checkpoint_dir, 'generated_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find generated_data.csv in {checkpoint_dir}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {checkpoint_dir}")
    return df


def plot_property_comparison(df1, df2, prop_name, label1, label2, output_dir):
    """Plot KDE comparison for a single property."""
    pred_col = f'Predicted {prop_name}'
    
    if pred_col not in df1.columns or pred_col not in df2.columns:
        print(f"Skipping {prop_name}: column not found in one or both datasets")
        return
    
    # Filter out NaN values
    data1 = df1[pred_col].dropna()
    data2 = df2[pred_col].dropna()
    
    if len(data1) == 0 or len(data2) == 0:
        print(f"Skipping {prop_name}: no valid data in one or both datasets")
        return
    
    plt.figure(figsize=(12, 7))
    
    # Plot KDE for both datasets
    sns.kdeplot(data=data1, label=label1, fill=True, alpha=0.5, linewidth=2, bw_adjust=2)
    sns.kdeplot(data=data2, label=label2, fill=True, alpha=0.5, linewidth=2, bw_adjust=2)
    
    # Formatting
    prop_display_names = {
        'logps': 'LogP',
        'qeds': 'QED',
        'sas': 'SAS',
        'tpsas': 'TPSA',
        'affinity': 'Binding Affinity'
    }
    
    prop_display = prop_display_names.get(prop_name, prop_name)
    plt.title(f'{prop_display} Distribution Comparison', fontsize=14, fontweight='bold')
    plt.xlabel(prop_display, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, f'{prop_name}_comparison_kde.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prop_name}_comparison_kde.png")


def plot_property_comparison_with_targets(df1, df2, prop_name, label1, label2, output_dir):
    """Plot KDE comparison for a single property, all target values on one graph."""
    target_col = f'Target {prop_name}'
    pred_col = f'Predicted {prop_name}'
    
    if target_col not in df1.columns or target_col not in df2.columns:
        print(f"Skipping target-based plot for {prop_name}: target column not found")
        return
    
    if pred_col not in df1.columns or pred_col not in df2.columns:
        print(f"Skipping target-based plot for {prop_name}: predicted column not found")
        return
    
    # Get common target values
    targets1 = set(df1[target_col].dropna().unique())
    targets2 = set(df2[target_col].dropna().unique())
    common_targets = sorted(targets1.intersection(targets2))
    
    if len(common_targets) == 0:
        print(f"Skipping target-based plot for {prop_name}: no common target values")
        return
    
    # Create a single plot for all targets
    plt.figure(figsize=(14, 8))
    
    # Define colors for different targets
    colors = plt.cm.tab10(np.linspace(0, 1, len(common_targets)))
    # Plot for each target value
    for idx, target_val in enumerate(common_targets):
        data1 = df1[df1[target_col] == target_val][pred_col].dropna()
        data2 = df2[df2[target_col] == target_val][pred_col].dropna()
        
        if len(data1) < 5 or len(data2) < 5:
            continue
        
        # Plot checkpoint 1 with solid line
        sns.kdeplot(data=data1, 
                   label=f'{label1} (Target={target_val:.2f})', 
                   fill=True, alpha=0.3, linewidth=2.5, 
                   linestyle='-',
                   color=colors[idx], bw_adjust=2)
        
        # Plot checkpoint 2 with dashed line (same color, different style)
        sns.kdeplot(data=data2, 
                   label=f'{label2} (Target={target_val:.2f})', 
                   fill=True, alpha=0.2, linewidth=2.5, 
                   linestyle='--',
                   color=colors[idx], bw_adjust=2)
        
        # Add vertical red line at target value
        plt.axvline(x=target_val, color='red', linestyle='-', linewidth=2, alpha=0.7)
    
    # Add a single legend entry for target lines
    plt.axvline(x=float('nan'), color='red', linestyle='-', linewidth=2, 
               alpha=0.7, label='Target Values')
    
    # Formatting
    prop_display_names = {
        'logps': 'LogP',
        'qeds': 'QED',
        'sas': 'SAS',
        'tpsas': 'TPSA',
        'affinity': 'Binding Affinity'
    }
    
    prop_display = prop_display_names.get(prop_name, prop_name)
    plt.title(f'{prop_display} Distribution by Target Value (All Targets)', 
             fontsize=14, fontweight='bold')
    plt.xlabel(prop_display, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=9, ncol=2, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, f'{prop_name}_all_targets_comparison_kde.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved target-specific comparison plot for {prop_name}")

def compute_summary_statistics(df1, df2, prop_name, label1, label2):
    """Compute and display summary statistics for comparison."""
    pred_col = f'Predicted {prop_name}'
    
    if pred_col not in df1.columns or pred_col not in df2.columns:
        return
    
    data1 = df1[pred_col].dropna()
    data2 = df2[pred_col].dropna()
    
    if len(data1) == 0 or len(data2) == 0:
        return
    
    print(f"\n{prop_name.upper()} Statistics:")
    print(f"{label1}:")
    print(f"  Mean: {data1.mean():.4f}, Std: {data1.std():.4f}")
    print(f"  Min: {data1.min():.4f}, Max: {data1.max():.4f}")
    print(f"  Median: {data1.median():.4f}")
    
    print(f"{label2}:")
    print(f"  Mean: {data2.mean():.4f}, Std: {data2.std():.4f}")
    print(f"  Min: {data2.min():.4f}, Max: {data2.max():.4f}")
    print(f"  Median: {data2.median():.4f}")


def create_summary_table(df1, df2, properties, label1, label2, output_dir):
    """Create a summary statistics table comparing both checkpoints."""
    summary_data = []
    
    for prop in properties:
        pred_col = f'Predicted {prop}'
        
        if pred_col not in df1.columns or pred_col not in df2.columns:
            continue
        
        data1 = df1[pred_col].dropna()
        data2 = df2[pred_col].dropna()
        
        if len(data1) == 0 or len(data2) == 0:
            continue
        
        summary_data.append({
            'Property': prop,
            f'{label1} Mean': data1.mean(),
            f'{label1} Std': data1.std(),
            f'{label1} Median': data1.median(),
            f'{label2} Mean': data2.mean(),
            f'{label2} Std': data2.std(),
            f'{label2} Median': data2.median(),
            'Mean Diff': data2.mean() - data1.mean(),
            'Std Diff': data2.std() - data1.std()
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        output_path = os.path.join(output_dir, 'comparison_summary.csv')
        summary_df.to_csv(output_path, index=False)
        print(f"\nSaved summary statistics to: comparison_summary.csv")
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))


def main():
    checkpoint_dir1 = args.checkpoint_dir1
    checkpoint_dir2 = args.checkpoint_dir2
    properties = args.properties
    label1 = args.label1
    label2 = args.label2
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(checkpoint_dir1, 'comparison')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print(f"Comparing checkpoints:")
    print(f"  {label1}: {checkpoint_dir1}")
    print(f"  {label2}: {checkpoint_dir2}")
    print(f"Properties: {properties}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Load data from both checkpoints
    df1 = load_checkpoint_data(checkpoint_dir1)
    df2 = load_checkpoint_data(checkpoint_dir2)
    
    # Generate comparison plots for each property
    print("\nGenerating comparison KDE plots...")
    for prop in properties:
        plot_property_comparison(df1, df2, prop, label1, label2, output_dir)
        compute_summary_statistics(df1, df2, prop, label1, label2)
    
    # Generate target-specific comparison plots
    print("\nGenerating target-specific comparison plots...")
    for prop in properties:
        plot_property_comparison_with_targets(df1, df2, prop, label1, label2, output_dir)
    
    # Create summary statistics table
    create_summary_table(df1, df2, properties, label1, label2, output_dir)
    
    print("\n" + "="*60)
    print(f"All comparison plots saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
