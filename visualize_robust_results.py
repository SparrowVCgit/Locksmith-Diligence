#!/usr/bin/env python3
"""
Visualization Script for Robust AlphaFold Analysis Results
=========================================================
Creates comprehensive visualizations for structural comparison results
including control baseline analysis and biological significance assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load analysis results and control data"""
    try:
        results = pd.read_csv('robust_analysis_results.csv')
        control = pd.read_csv('robust_analysis_control.csv')
        return results, control
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def create_rmsd_comparison_plot(results, control):
    """Create RMSD comparison plot with control baseline"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: RMSD distribution with control baseline
    rmsd_values = results['rmsd'].values
    control_rmsds = control['rmsd'].values
    
    # Control statistics
    control_max = control_rmsds.max()
    control_mean = control_rmsds.mean()
    control_std = control_rmsds.std()
    
    # Create violin plot for variants
    ax1.violinplot([rmsd_values], positions=[1], widths=0.7, showmeans=True)
    
    # Add control distribution
    ax1.violinplot([control_rmsds], positions=[0], widths=0.7, showmeans=True)
    
    # Add threshold lines
    ax1.axhline(y=control_max, color='red', linestyle='--', alpha=0.7, 
                label=f'Control Max: {control_max:.1f}Å')
    ax1.axhline(y=control_mean, color='blue', linestyle='-', alpha=0.7,
                label=f'Control Mean: {control_mean:.1f}Å')
    ax1.axhline(y=control_mean + control_std, color='orange', linestyle=':', alpha=0.7,
                label=f'Mean + 1SD: {control_mean + control_std:.1f}Å')
    ax1.axhline(y=control_mean + 2*control_std, color='purple', linestyle=':', alpha=0.7,
                label=f'Mean + 2SD: {control_mean + 2*control_std:.1f}Å')
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Control\n(WT vs WT)', 'Variants\n(WT vs Mutants)'])
    ax1.set_ylabel('RMSD (Å)')
    ax1.set_title('RMSD Distribution: Control vs Variants')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual variant RMSDs with significance coloring
    results_sorted = results.sort_values('rmsd')
    
    colors = []
    for _, row in results_sorted.iterrows():
        if 'Potentially significant' in row['biological_significance']:
            colors.append('red')
        elif 'Above control mean + 2SD' in row['biological_significance']:
            colors.append('orange')
        elif 'Above control mean + 1SD' in row['biological_significance']:
            colors.append('yellow')
        else:
            colors.append('lightblue')
    
    bars = ax2.bar(range(len(results_sorted)), results_sorted['rmsd'], color=colors, alpha=0.7)
    
    # Add threshold lines
    ax2.axhline(y=control_max, color='red', linestyle='--', alpha=0.7, 
                label=f'Control Baseline: {control_max:.1f}Å')
    ax2.axhline(y=control_mean, color='blue', linestyle='-', alpha=0.7,
                label=f'Control Mean: {control_mean:.1f}Å')
    
    ax2.set_xlabel('Variants (sorted by RMSD)')
    ax2.set_ylabel('RMSD (Å)')
    ax2.set_title('Individual Variant RMSDs vs Control Baseline')
    ax2.set_xticks(range(len(results_sorted)))
    ax2.set_xticklabels([name.replace('_p53_', '_') for name in results_sorted['mobile']], 
                        rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('robust_rmsd_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confidence_analysis_plot(results):
    """Create plot showing relationship between confidence and RMSD"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: RMSD vs Mean pLDDT
    scatter = ax1.scatter(results['mean_plddt'], results['rmsd'], 
                         c=results['rmsd'], cmap='viridis', alpha=0.7, s=100)
    
    ax1.set_xlabel('Mean pLDDT Score')
    ax1.set_ylabel('RMSD (Å)')
    ax1.set_title('RMSD vs Mean pLDDT Confidence')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('RMSD (Å)')
    
    # Plot 2: Confidence mask percentage vs RMSD
    ax2.scatter(results['confidence_mask_pct'], results['rmsd'], alpha=0.7, s=100)
    
    ax2.set_xlabel('Confidence Mask Percentage (%)')
    ax2.set_ylabel('RMSD (Å)')
    ax2.set_title('RMSD vs Low-Confidence Regions (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(results['confidence_mask_pct'], results['rmsd'], 1)
    p = np.poly1d(z)
    ax2.plot(results['confidence_mask_pct'], p(results['confidence_mask_pct']), 
             "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.1f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('robust_confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_gdt_analysis_plot(results):
    """Create GDT-TS analysis plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: GDT-TS distribution
    ax1.hist(results['gdt_ts'], bins=15, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('GDT-TS Score (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of GDT-TS Scores')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    gdt_mean = results['gdt_ts'].mean()
    gdt_std = results['gdt_ts'].std()
    ax1.axvline(gdt_mean, color='red', linestyle='--', 
                label=f'Mean: {gdt_mean:.1f}%')
    ax1.legend()
    
    # Plot 2: RMSD vs GDT-TS relationship
    scatter = ax2.scatter(results['gdt_ts'], results['rmsd'], 
                         c=results['rmsd'], cmap='viridis_r', alpha=0.7, s=100)
    
    ax2.set_xlabel('GDT-TS Score (%)')
    ax2.set_ylabel('RMSD (Å)')
    ax2.set_title('RMSD vs GDT-TS Relationship')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    if len(results['gdt_ts']) > 1:
        z = np.polyfit(results['gdt_ts'], results['rmsd'], 1)
        p = np.poly1d(z)
        ax2.plot(results['gdt_ts'], p(results['gdt_ts']), 
                 "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.1f}')
        ax2.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('RMSD (Å)')
    
    plt.tight_layout()
    plt.savefig('robust_gdt_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_significance_summary_plot(results):
    """Create summary plot of biological significance categories"""
    
    # Count variants in each category
    significance_counts = {}
    for _, row in results.iterrows():
        sig = row['biological_significance']
        if 'Potentially significant' in sig:
            key = 'Potentially\nSignificant'
        elif 'Above control mean + 2SD' in sig:
            key = 'Above Mean\n+ 2SD'
        elif 'Above control mean + 1SD' in sig:
            key = 'Above Mean\n+ 1SD'
        else:
            key = 'Within Control\nVariability'
        
        significance_counts[key] = significance_counts.get(key, 0) + 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart of significance categories
    categories = list(significance_counts.keys())
    counts = list(significance_counts.values())
    colors = ['red', 'orange', 'yellow', 'lightblue'][:len(categories)]
    
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Variants')
    ax1.set_title('Biological Significance Categories')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Pie chart
    ax2.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', 
            startangle=90, explode=[0.1 if 'Potentially' in cat else 0 for cat in categories])
    ax2.set_title('Distribution of Biological Significance')
    
    plt.tight_layout()
    plt.savefig('robust_significance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_variant_comparison(results):
    """Create detailed comparison of specific variants"""
    
    # Get potentially significant variants
    significant = results[results['biological_significance'].str.contains('Potentially significant')]
    
    if len(significant) == 0:
        print("No potentially significant variants found.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: RMSD comparison
    all_variants = results['mobile'].str.replace('_p53_', '_')
    sig_variants = significant['mobile'].str.replace('_p53_', '_')
    
    colors = ['red' if var in sig_variants.values else 'lightblue' for var in all_variants]
    
    axes[0,0].bar(range(len(results)), results['rmsd'], color=colors, alpha=0.7)
    axes[0,0].set_title('RMSD by Variant (Red = Potentially Significant)')
    axes[0,0].set_ylabel('RMSD (Å)')
    axes[0,0].set_xticks(range(len(results)))
    axes[0,0].set_xticklabels(all_variants, rotation=45, ha='right')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: GDT-TS comparison
    axes[0,1].bar(range(len(results)), results['gdt_ts'], color=colors, alpha=0.7)
    axes[0,1].set_title('GDT-TS by Variant')
    axes[0,1].set_ylabel('GDT-TS (%)')
    axes[0,1].set_xticks(range(len(results)))
    axes[0,1].set_xticklabels(all_variants, rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Confidence comparison
    axes[1,0].bar(range(len(results)), results['mean_plddt'], color=colors, alpha=0.7)
    axes[1,0].set_title('Mean pLDDT by Variant')
    axes[1,0].set_ylabel('Mean pLDDT')
    axes[1,0].set_xticks(range(len(results)))
    axes[1,0].set_xticklabels(all_variants, rotation=45, ha='right')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Coverage comparison
    axes[1,1].bar(range(len(results)), results['coverage'], color=colors, alpha=0.7)
    axes[1,1].set_title('Alignment Coverage by Variant')
    axes[1,1].set_ylabel('Coverage (%)')
    axes[1,1].set_xticks(range(len(results)))
    axes[1,1].set_xticklabels(all_variants, rotation=45, ha='right')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('robust_detailed_variant_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed info for significant variants
    print("\n=== POTENTIALLY SIGNIFICANT VARIANTS ===")
    for _, row in significant.iterrows():
        print(f"\nVariant: {row['mobile']}")
        print(f"  RMSD: {row['rmsd']:.2f}Å")
        print(f"  GDT-TS: {row['gdt_ts']:.1f}%")
        print(f"  Mean pLDDT: {row['mean_plddt']:.1f}")
        print(f"  Coverage: {row['coverage']:.1f}%")
        print(f"  Confidence Mask: {row['confidence_mask_pct']:.1f}%")

def main():
    """Main function to generate all visualizations"""
    print("Loading robust AlphaFold analysis results...")
    
    results, control = load_data()
    if results is None or control is None:
        return
    
    print(f"Loaded {len(results)} variant comparisons and {len(control)} control comparisons")
    
    # Generate all visualizations
    print("\n1. Creating RMSD comparison plots...")
    create_rmsd_comparison_plot(results, control)
    
    print("2. Creating confidence analysis plots...")
    create_confidence_analysis_plot(results)
    
    print("3. Creating GDT-TS analysis plots...")
    create_gdt_analysis_plot(results)
    
    print("4. Creating significance summary plots...")
    create_significance_summary_plot(results)
    
    print("5. Creating detailed variant comparison...")
    create_detailed_variant_comparison(results)
    
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Total variants analyzed: {len(results)}")
    
    # Control statistics
    control_max = control['rmsd'].max()
    control_mean = control['rmsd'].mean()
    control_std = control['rmsd'].std()
    
    print(f"Control baseline (max RMSD): {control_max:.2f}Å")
    print(f"Control mean ± SD: {control_mean:.2f} ± {control_std:.2f}Å")
    
    # Significance counts
    significant_count = len(results[results['biological_significance'].str.contains('Potentially significant')])
    print(f"Potentially significant variants: {significant_count}")
    print(f"Within control variability: {len(results) - significant_count}")
    
    print("\nAll visualizations saved as PNG files!")

if __name__ == "__main__":
    main() 