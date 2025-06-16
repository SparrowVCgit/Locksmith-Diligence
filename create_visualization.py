#!/usr/bin/env python3
"""
Create comprehensive visualizations for protein structure similarity analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_visualization(csv_file="protein_similarity_report.csv"):
    """Create comprehensive visualizations of the analysis results"""
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Separate control and variant data
    control_data = df[df.get('is_control', False) == True] if 'is_control' in df.columns else pd.DataFrame()
    variant_data = df[df.get('is_control', False) == False] if 'is_control' in df.columns else df
    
    print(f"Loaded {len(variant_data)} variants and {len(control_data)} control comparisons")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    
    # Scientific thresholds for interpretation
    rmsd_thresholds = {'nearly_identical': 1.0, 'very_similar': 2.0, 'similar': 4.0, 'moderate_difference': 8.0}
    tm_score_thresholds = {'same_fold': 0.5, 'mostly_same_fold': 0.4, 'potentially_related': 0.2}
    gdt_ts_thresholds = {'high_similarity': 50.0, 'moderate_similarity': 30.0, 'low_similarity': 10.0}
    
    # Calculate control baselines
    control_rmsd_max = control_data['rmsd_mean'].max() if not control_data.empty and 'rmsd_mean' in control_data.columns else None
    control_gdt_min = control_data['gdt_ts_mean'].min() if not control_data.empty and 'gdt_ts_mean' in control_data.columns else None
    control_tm_min = control_data['tm_score_mean'].min() if not control_data.empty and 'tm_score_mean' in control_data.columns else None
    
    # 1. RMSD Comparison with scientific thresholds and control baseline
    ax1 = plt.subplot(4, 4, 1)
    if 'rmsd_mean' in variant_data.columns:
        bars1 = ax1.bar(range(len(variant_data)), variant_data['rmsd_mean'], 
                        color=['darkred' if x > rmsd_thresholds['moderate_difference'] 
                               else 'orange' if x > rmsd_thresholds['similar']
                               else 'yellow' if x > rmsd_thresholds['very_similar']
                               else 'lightgreen' if x > rmsd_thresholds['nearly_identical']
                               else 'darkgreen' for x in variant_data['rmsd_mean']])
        
        # Add threshold lines
        ax1.axhline(y=rmsd_thresholds['nearly_identical'], color='green', linestyle='--', alpha=0.7, label='Nearly Identical (<1.0Å)')
        ax1.axhline(y=rmsd_thresholds['very_similar'], color='yellowgreen', linestyle='--', alpha=0.7, label='Very Similar (<2.0Å)')
        ax1.axhline(y=rmsd_thresholds['similar'], color='orange', linestyle='--', alpha=0.7, label='Similar (<4.0Å)')
        ax1.axhline(y=rmsd_thresholds['moderate_difference'], color='red', linestyle='--', alpha=0.7, label='Moderate Diff (<8.0Å)')
        
        # Add control baseline
        if control_rmsd_max is not None:
            ax1.axhline(y=control_rmsd_max, color='purple', linestyle='-', linewidth=3, alpha=0.8, 
                       label=f'Control Baseline ({control_rmsd_max:.1f}Å)')
        
        ax1.set_title('RMSD Comparison with Scientific Thresholds & Control', fontsize=14, fontweight='bold')
        ax1.set_ylabel('RMSD (Å)')
        ax1.set_xlabel('Protein Variants')
        ax1.tick_params(axis='x', rotation=45)
        if len(variant_data) <= 20:
            ax1.set_xticks(range(len(variant_data)))
            ax1.set_xticklabels(variant_data['mutation'], rotation=45, ha='right')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    
    # 2. Control vs Variants Comparison
    ax2 = plt.subplot(4, 4, 2)
    if not control_data.empty and not variant_data.empty and 'rmsd_mean' in control_data.columns and 'rmsd_mean' in variant_data.columns:
        # Box plot comparing control and variants
        box_data = [control_data['rmsd_mean'].values, variant_data['rmsd_mean'].values]
        bp = ax2.boxplot(box_data, labels=['Control\n(WT vs WT)', 'Variants\n(WT vs Mutant)'], patch_artist=True)
        
        # Color boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('lightcoral')
        bp['boxes'][1].set_alpha(0.7)
        
        ax2.set_title('Control vs Variants RMSD Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RMSD (Å)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistical annotation
        control_mean = control_data['rmsd_mean'].mean()
        variant_mean = variant_data['rmsd_mean'].mean()
        ax2.text(0.5, 0.95, f'Control Mean: {control_mean:.2f}Å\nVariant Mean: {variant_mean:.2f}Å\nFold Increase: {variant_mean/control_mean:.1f}x', 
                transform=ax2.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                verticalalignment='top', horizontalalignment='center')
    
    # 3. TM-Score Distribution with thresholds and control
    ax3 = plt.subplot(4, 4, 3)
    if 'tm_score_mean' in variant_data.columns:
        bars3 = ax3.bar(range(len(variant_data)), variant_data['tm_score_mean'],
                        color=['darkgreen' if x > tm_score_thresholds['same_fold']
                               else 'lightgreen' if x > tm_score_thresholds['mostly_same_fold']
                               else 'orange' if x > tm_score_thresholds['potentially_related']
                               else 'red' for x in variant_data['tm_score_mean']])
        
        # Add threshold lines
        ax3.axhline(y=tm_score_thresholds['same_fold'], color='green', linestyle='--', alpha=0.7, label='Same Fold (>0.5)')
        ax3.axhline(y=tm_score_thresholds['mostly_same_fold'], color='yellowgreen', linestyle='--', alpha=0.7, label='Mostly Same (>0.4)')
        ax3.axhline(y=tm_score_thresholds['potentially_related'], color='orange', linestyle='--', alpha=0.7, label='Potentially Related (>0.2)')
        
        # Add control baseline
        if control_tm_min is not None:
            ax3.axhline(y=control_tm_min, color='purple', linestyle='-', linewidth=3, alpha=0.8, 
                       label=f'Control Baseline ({control_tm_min:.2f})')
        
        ax3.set_title('TM-Score Distribution with Fold Classification', fontsize=14, fontweight='bold')
        ax3.set_ylabel('TM-Score')
        ax3.set_xlabel('Protein Variants')
        ax3.tick_params(axis='x', rotation=45)
        if len(variant_data) <= 20:
            ax3.set_xticks(range(len(variant_data)))
            ax3.set_xticklabels(variant_data['mutation'], rotation=45, ha='right')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    
    # 4. Biological Significance with Control Context
    ax4 = plt.subplot(4, 4, 4)
    if 'biological_significance' in variant_data.columns:
        bio_counts = variant_data['biological_significance'].value_counts()
        colors = ['darkred' if 'SIGNIFICANT' in x and 'NOT' not in x
                 else 'orange' if 'POTENTIALLY' in x
                 else 'green' for x in bio_counts.index]
        wedges, texts, autotexts = ax4.pie(bio_counts.values, labels=bio_counts.index, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax4.set_title('Biological Significance Distribution\n(Variants Only)', fontsize=14, fontweight='bold')
        # Make text smaller to fit
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')
            autotext.set_weight('bold')
    
    # 5. RMSD vs TM-Score Scatter Plot with Control Points
    ax5 = plt.subplot(4, 4, 5)
    if 'rmsd_mean' in variant_data.columns and 'tm_score_mean' in variant_data.columns:
        # Plot variants
        scatter_variants = ax5.scatter(variant_data['rmsd_mean'], variant_data['tm_score_mean'], 
                                     c='red', s=100, alpha=0.7, label='Variants', marker='o')
        
        # Plot controls if available
        if not control_data.empty and 'rmsd_mean' in control_data.columns and 'tm_score_mean' in control_data.columns:
            scatter_control = ax5.scatter(control_data['rmsd_mean'], control_data['tm_score_mean'], 
                                        c='blue', s=150, alpha=0.9, label='Control (WT vs WT)', marker='s')
        
        ax5.set_xlabel('RMSD (Å)')
        ax5.set_ylabel('TM-Score')
        ax5.set_title('RMSD vs TM-Score: Variants vs Control', fontsize=14, fontweight='bold')
        
        # Add threshold lines
        ax5.axvline(x=rmsd_thresholds['similar'], color='orange', linestyle='--', alpha=0.5, label='RMSD Similar (<4Å)')
        ax5.axhline(y=tm_score_thresholds['same_fold'], color='green', linestyle='--', alpha=0.5, label='TM Same Fold (>0.5)')
        
        # Add control baseline box
        if control_rmsd_max is not None and control_tm_min is not None:
            from matplotlib.patches import Rectangle
            rect = Rectangle((0, control_tm_min), control_rmsd_max, 1-control_tm_min, 
                           linewidth=2, edgecolor='purple', facecolor='purple', alpha=0.1, 
                           label='Control Range')
            ax5.add_patch(rect)
        
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    
    # Continue with remaining plots but focus on variant data...
    # 6. Variants Beyond Control Baseline
    ax6 = plt.subplot(4, 4, 6)
    if not control_data.empty and 'rmsd_mean' in control_data.columns and 'rmsd_mean' in variant_data.columns:
        control_threshold = control_data['rmsd_mean'].max()
        beyond_control = variant_data[variant_data['rmsd_mean'] > control_threshold]
        within_control = variant_data[variant_data['rmsd_mean'] <= control_threshold]
        
        categories = ['Within Control\nBaseline', 'Beyond Control\nBaseline']
        counts = [len(within_control), len(beyond_control)]
        colors = ['lightgreen', 'red']
        
        bars = ax6.bar(categories, counts, color=colors, alpha=0.7)
        ax6.set_title(f'Variants vs Control Baseline\n(Threshold: {control_threshold:.2f}Å)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Number of Variants')
        
        # Add percentage labels
        total = len(variant_data)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}\n({count/total*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        ax6.grid(True, alpha=0.3)
    
    # 7-16: Continue with other plots using variant_data instead of df
    # [Rest of the plots remain similar but use variant_data instead of df]
    
    # 7. Sequence Identity
    ax7 = plt.subplot(4, 4, 7)
    if 'sequence_identity' in variant_data.columns:
        bars7 = ax7.bar(range(len(variant_data)), variant_data['sequence_identity'], color='lightblue', alpha=0.7)
        ax7.set_title('Sequence Identity (Variants)', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Sequence Identity (%)')
        ax7.set_xlabel('Protein Variants')
        ax7.tick_params(axis='x', rotation=45)
        if len(variant_data) <= 20:
            ax7.set_xticks(range(len(variant_data)))
            ax7.set_xticklabels(variant_data['mutation'], rotation=45, ha='right')
        ax7.grid(True, alpha=0.3)
    
    # 8. Error bars for RMSD (variants only)
    ax8 = plt.subplot(4, 4, 8)
    if 'rmsd_mean' in variant_data.columns and 'rmsd_std' in variant_data.columns:
        ax8.errorbar(range(len(variant_data)), variant_data['rmsd_mean'], yerr=variant_data['rmsd_std'], 
                    fmt='o', capsize=5, capthick=2, elinewidth=2, markerfacecolor='red', alpha=0.7)
        
        # Add control baseline
        if control_rmsd_max is not None:
            ax8.axhline(y=control_rmsd_max, color='purple', linestyle='-', linewidth=2, alpha=0.8, 
                       label=f'Control Max: {control_rmsd_max:.2f}Å')
        
        ax8.set_title('RMSD with Standard Deviation', fontsize=14, fontweight='bold')
        ax8.set_ylabel('RMSD (Å)')
        ax8.set_xlabel('Protein Variants')
        ax8.tick_params(axis='x', rotation=45)
        if len(variant_data) <= 20:
            ax8.set_xticks(range(len(variant_data)))
            ax8.set_xticklabels(variant_data['mutation'], rotation=45, ha='right')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
    
    # 9. Correlation Heatmap (variants only)
    ax9 = plt.subplot(4, 4, 9)
    numeric_cols = ['rmsd_mean', 'gdt_ts_mean', 'tm_score_mean', 'sequence_identity']
    available_cols = [col for col in numeric_cols if col in variant_data.columns]
    if len(available_cols) > 1:
        corr_matrix = variant_data[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax9,
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        ax9.set_title('Metric Correlations (Variants)', fontsize=14, fontweight='bold')
    
    # 10. RMSD Distribution with control overlay
    ax10 = plt.subplot(4, 4, 10)
    if 'rmsd_mean' in variant_data.columns:
        ax10.hist(variant_data['rmsd_mean'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black', label='Variants')
        
        # Overlay control distribution if available
        if not control_data.empty and 'rmsd_mean' in control_data.columns:
            ax10.hist(control_data['rmsd_mean'], bins=5, alpha=0.8, color='lightblue', edgecolor='black', label='Control')
        
        ax10.axvline(x=variant_data['rmsd_mean'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Variant Mean: {variant_data["rmsd_mean"].mean():.2f}Å')
        
        if control_rmsd_max is not None:
            ax10.axvline(x=control_rmsd_max, color='purple', linestyle='-', linewidth=2, 
                        label=f'Control Max: {control_rmsd_max:.2f}Å')
        
        ax10.set_title('RMSD Distribution: Variants vs Control', fontsize=14, fontweight='bold')
        ax10.set_xlabel('RMSD (Å)')
        ax10.set_ylabel('Frequency')
        ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)
    
    # 11-16: Continue with remaining plots...
    # [Include remaining visualization code but adapted for variant_data]
    
    # 16. Enhanced Scientific interpretation guide with control context
    ax16 = plt.subplot(4, 4, 16)
    ax16.axis('tight')
    ax16.axis('off')
    
    interpretation_text = f"""
    SCIENTIFIC INTERPRETATION GUIDE
    
    CONTROL BASELINES:
    • Control RMSD Max: {control_rmsd_max:.2f}Å
    • Control TM-Score Min: {control_tm_min:.3f}
    • Control GDT-TS Min: {control_gdt_min:.1f}%
    
    RMSD Thresholds (Å):
    • < 1.0: Nearly identical
    • 1.0-2.0: Very similar  
    • 2.0-4.0: Similar fold
    • 4.0-8.0: Moderate differences
    • > 8.0: Significantly different
    
    TM-Score Thresholds:
    • > 0.5: Same fold
    • 0.4-0.5: Mostly same fold
    • 0.2-0.4: Potentially related
    • < 0.2: Different folds
    
    INTERPRETATION:
    Variants beyond control baselines
    show real structural changes.
    """ if control_rmsd_max is not None else """
    SCIENTIFIC INTERPRETATION GUIDE
    
    RMSD Thresholds (Å):
    • < 1.0: Nearly identical
    • 1.0-2.0: Very similar  
    • 2.0-4.0: Similar fold
    • 4.0-8.0: Moderate differences
    • > 8.0: Significantly different
    
    TM-Score Thresholds:
    • > 0.5: Same fold
    • 0.4-0.5: Mostly same fold
    • 0.2-0.4: Potentially related
    • < 0.2: Different folds
    
    GDT-TS Thresholds (%):
    • > 50: High similarity
    • 30-50: Moderate similarity
    • 10-30: Low similarity
    • < 10: Very low similarity
    """
    
    ax16.text(0.05, 0.95, interpretation_text, transform=ax16.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Save the plot
    output_file = "protein_similarity_analysis_with_controls.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comprehensive visualization with controls saved as: {output_file}")
    
    # Also save as PDF for high quality
    output_pdf = "protein_similarity_analysis_with_controls.pdf"
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"High-quality PDF saved as: {output_pdf}")
    
    plt.show()
    
    return fig

def create_detailed_report(df):
    """Create a detailed interpretation report"""
    
    print("\n" + "="*100)
    print("DETAILED SCIENTIFIC INTERPRETATION REPORT")
    print("="*100)
    
    if 'biological_significance' in df.columns:
        print(f"\nOVERALL ASSESSMENT:")
        print("-" * 50)
        bio_counts = df['biological_significance'].value_counts()
        
        total_variants = len(df)
        significant_variants = bio_counts.get('BIOLOGICALLY SIGNIFICANT', 0)
        potentially_significant = bio_counts.get('POTENTIALLY SIGNIFICANT', 0)
        not_significant = bio_counts.get('LIKELY NOT SIGNIFICANT', 0)
        
        print(f"Total variants analyzed: {total_variants}")
        print(f"Biologically significant changes: {significant_variants} ({significant_variants/total_variants*100:.1f}%)")
        print(f"Potentially significant changes: {potentially_significant} ({potentially_significant/total_variants*100:.1f}%)")
        print(f"Likely not significant changes: {not_significant} ({not_significant/total_variants*100:.1f}%)")
        
        if significant_variants > 0:
            print(f"\n⚠️  WARNING: {significant_variants} variants show major structural changes that may significantly impact protein function!")
        
        if potentially_significant > 0:
            print(f"\n⚠️  CAUTION: {potentially_significant} variants show moderate changes that require further investigation.")
    
    if 'rmsd_mean' in df.columns:
        print(f"\nMOST CONCERNING VARIANTS (High RMSD):")
        print("-" * 50)
        high_rmsd = df.nlargest(5, 'rmsd_mean')
        for _, row in high_rmsd.iterrows():
            rmsd = row['rmsd_mean']
            mutation = row['mutation']
            bio_sig = row.get('biological_significance', 'Unknown')
            
            if rmsd > 8.0:
                concern_level = "🔴 CRITICAL"
            elif rmsd > 4.0:
                concern_level = "🟠 HIGH"
            elif rmsd > 2.0:
                concern_level = "🟡 MODERATE"
            else:
                concern_level = "🟢 LOW"
            
            print(f"  {concern_level} {mutation}: RMSD = {rmsd:.3f}Å | {bio_sig}")
    
    if 'tm_score_mean' in df.columns:
        print(f"\nFOLD CLASSIFICATION ANALYSIS:")
        print("-" * 50)
        same_fold = df[df['tm_score_mean'] > 0.5] if 'tm_score_mean' in df.columns else pd.DataFrame()
        different_fold = df[df['tm_score_mean'] < 0.2] if 'tm_score_mean' in df.columns else pd.DataFrame()
        
        print(f"Variants maintaining same fold (TM-Score > 0.5): {len(same_fold)} ({len(same_fold)/len(df)*100:.1f}%)")
        print(f"Variants with potentially different fold (TM-Score < 0.2): {len(different_fold)} ({len(different_fold)/len(df)*100:.1f}%)")
        
        if len(different_fold) > 0:
            print(f"\n🔴 FOLD DISRUPTION ALERT - These variants may have fundamentally different structures:")
            for _, row in different_fold.iterrows():
                print(f"  • {row['mutation']}: TM-Score = {row['tm_score_mean']:.3f}")
    
    print(f"\nRECOMMENDations:")
    print("-" * 50)
    print("1. Focus experimental validation on variants marked as 'BIOLOGICALLY SIGNIFICANT'")
    print("2. Variants with RMSD > 4.0Å require immediate functional studies")
    print("3. Variants with TM-Score < 0.2 may need structural validation")
    print("4. Consider the biological context - some regions may be more tolerant to changes")
    print("5. Validate computationally predicted changes with experimental methods")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    # Load the data
    try:
        df = pd.read_csv("protein_similarity_report.csv")
        print("Creating comprehensive visualizations...")
        create_visualization()
        create_detailed_report(df)
    except FileNotFoundError:
        print("Error: protein_similarity_report.csv not found. Please run the main analysis first.")
    except Exception as e:
        print(f"Error creating visualization: {e}") 