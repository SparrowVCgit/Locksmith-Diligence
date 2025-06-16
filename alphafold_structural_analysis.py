#!/usr/bin/env python3
"""
AlphaFold Structural Difference Analysis - Robust Workflow
=========================================================
A comprehensive tool for analyzing structural differences from AlphaFold outputs
using biologically meaningful metrics and confidence-based filtering.

This implementation follows best practices for AlphaFold structure comparison:
1. Preprocessing with confidence filtering
2. Robust structural alignment 
3. Multiple complementary metrics
4. Proper visualization and interpretation
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import warnings
import subprocess
import tempfile
warnings.filterwarnings('ignore')

try:
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.mmcifio import MMCIFIO
    from Bio import BiopythonWarning
    warnings.simplefilter('ignore', BiopythonWarning)
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. Install with: pip install biopython")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Plotting libraries not available. Install with: pip install matplotlib seaborn")

class ConfidenceBasedSelect(Select):
    """Select atoms based on confidence scores (pLDDT)"""
    
    def __init__(self, confidence_threshold=70.0, confidence_dict=None):
        self.confidence_threshold = confidence_threshold
        self.confidence_dict = confidence_dict or {}
    
    def accept_residue(self, residue):
        """Accept residue if confidence is above threshold"""
        if not self.confidence_dict:
            return True  # Accept all if no confidence data
        
        res_id = residue.get_id()[1]  # Get residue number
        confidence = self.confidence_dict.get(res_id, 0)
        return confidence >= self.confidence_threshold

class AlphaFoldStructuralAnalyzer:
    """Comprehensive AlphaFold structural difference analyzer"""
    
    def __init__(self, base_dir, confidence_threshold=70.0):
        self.base_dir = Path(base_dir)
        self.confidence_threshold = confidence_threshold
        self.wildtype_dir = self.base_dir / "p53_wildtype"
        self.results = []
        self.wildtype_structures = {}
        self.wildtype_confidences = {}
        
    def load_confidence_data(self, json_file):
        """Load confidence data from AlphaFold JSON file"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract per-residue confidence if available
            confidence_dict = {}
            if 'residue_confidences' in data:
                for i, conf in enumerate(data['residue_confidences'], 1):
                    confidence_dict[i] = conf
            elif 'plddt' in data:
                for i, conf in enumerate(data['plddt'], 1):
                    confidence_dict[i] = conf
            
            return data, confidence_dict
        except Exception as e:
            print(f"Error loading confidence data from {json_file}: {e}")
            return None, {}
    
    def convert_cif_to_pdb(self, cif_file, pdb_file, confidence_dict=None):
        """Convert CIF to PDB with optional confidence-based filtering"""
        if not BIOPYTHON_AVAILABLE:
            return False
        
        try:
            # Parse CIF file
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("protein", cif_file)
            
            # Create PDB writer
            io = PDBIO()
            io.set_structure(structure)
            
            # Apply confidence-based selection if confidence data available
            if confidence_dict:
                selector = ConfidenceBasedSelect(self.confidence_threshold, confidence_dict)
                io.save(str(pdb_file), selector)
            else:
                io.save(str(pdb_file))
            
            return True
        except Exception as e:
            print(f"Error converting {cif_file} to PDB: {e}")
            return False
    
    def run_tmalign(self, structure1_pdb, structure2_pdb):
        """Run TM-align for structure comparison"""
        try:
            # Check if TM-align is available
            result = subprocess.run(['which', 'TMalign'], capture_output=True, text=True)
            if result.returncode != 0:
                print("Warning: TM-align not found. Install TM-align for best results.")
                return None
            
            # Run TM-align
            cmd = ['TMalign', str(structure1_pdb), str(structure2_pdb)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"TM-align failed: {result.stderr}")
                return None
            
            # Parse TM-align output
            lines = result.stdout.strip().split('\n')
            tm_results = {}
            
            for line in lines:
                if 'TM-score=' in line and 'Chain_1' in line:
                    tm_score = float(line.split('TM-score=')[1].split('(')[0].strip())
                    tm_results['tm_score_1'] = tm_score
                elif 'TM-score=' in line and 'Chain_2' in line:
                    tm_score = float(line.split('TM-score=')[1].split('(')[0].strip())
                    tm_results['tm_score_2'] = tm_score
                elif 'RMSD of' in line:
                    rmsd = float(line.split('RMSD of')[1].split('common')[0].strip())
                    tm_results['rmsd'] = rmsd
                elif 'Aligned length=' in line:
                    aligned_length = int(line.split('Aligned length=')[1].split(',')[0].strip())
                    tm_results['aligned_length'] = aligned_length
            
            return tm_results
        except Exception as e:
            print(f"Error running TM-align: {e}")
            return None
    
    def calculate_rmsd_biopython(self, pdb1, pdb2):
        """Calculate RMSD using BioPython as fallback"""
        if not BIOPYTHON_AVAILABLE:
            return None
        
        try:
            from Bio.PDB import Superimposer
            
            parser = PDBParser(QUIET=True)
            structure1 = parser.get_structure("s1", pdb1)
            structure2 = parser.get_structure("s2", pdb2)
            
            # Get CA atoms
            ca_atoms_1 = []
            ca_atoms_2 = []
            
            for model in structure1:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            ca_atoms_1.append(residue['CA'])
            
            for model in structure2:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            ca_atoms_2.append(residue['CA'])
            
            # Use minimum length
            min_len = min(len(ca_atoms_1), len(ca_atoms_2))
            if min_len < 3:
                return None
            
            ca_atoms_1 = ca_atoms_1[:min_len]
            ca_atoms_2 = ca_atoms_2[:min_len]
            
            # Calculate RMSD
            superimposer = Superimposer()
            superimposer.set_atoms(ca_atoms_1, ca_atoms_2)
            
            return {
                'rmsd': superimposer.rms,
                'aligned_length': min_len,
                'tm_score_1': None,
                'tm_score_2': None
            }
        except Exception as e:
            print(f"Error calculating RMSD with BioPython: {e}")
            return None
    
    def calculate_gdt_ts(self, pdb1, pdb2, thresholds=[1, 2, 4, 8]):
        """Calculate GDT-TS score"""
        if not BIOPYTHON_AVAILABLE:
            return None
        
        try:
            from Bio.PDB import Superimposer
            
            parser = PDBParser(QUIET=True)
            structure1 = parser.get_structure("s1", pdb1)
            structure2 = parser.get_structure("s2", pdb2)
            
            # Get CA coordinates
            coords1, coords2 = [], []
            
            for model in structure1:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coords1.append(residue['CA'].coord)
            
            for model in structure2:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coords2.append(residue['CA'].coord)
            
            min_len = min(len(coords1), len(coords2))
            if min_len < 3:
                return None
            
            coords1 = np.array(coords1[:min_len])
            coords2 = np.array(coords2[:min_len])
            
            # Align structures
            from Bio.PDB.Atom import Atom
            atoms1 = [Atom("CA", coord, 0, 1, " ", "CA", i, "C") for i, coord in enumerate(coords1)]
            atoms2 = [Atom("CA", coord, 0, 1, " ", "CA", i, "C") for i, coord in enumerate(coords2)]
            
            superimposer = Superimposer()
            superimposer.set_atoms(atoms1, atoms2)
            superimposer.apply(atoms2)
            
            # Calculate distances after alignment
            aligned_coords2 = np.array([atom.coord for atom in atoms2])
            distances = np.linalg.norm(coords1 - aligned_coords2, axis=1)
            
            # Calculate GDT-TS
            gdt_scores = []
            for threshold in thresholds:
                within_threshold = np.sum(distances <= threshold)
                gdt_scores.append(within_threshold / len(distances))
            
            return np.mean(gdt_scores) * 100
        except Exception as e:
            print(f"Error calculating GDT-TS: {e}")
            return None
    
    def analyze_confidence_distribution(self, confidence_dict):
        """Analyze confidence score distribution"""
        if not confidence_dict:
            return {}
        
        confidences = list(confidence_dict.values())
        return {
            'mean_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'std_confidence': np.std(confidences),
            'high_confidence_fraction': np.sum(np.array(confidences) >= 90) / len(confidences),
            'medium_confidence_fraction': np.sum((np.array(confidences) >= 70) & (np.array(confidences) < 90)) / len(confidences),
            'low_confidence_fraction': np.sum(np.array(confidences) < 70) / len(confidences)
        }
    
    def load_wildtype_data(self):
        """Load and preprocess wildtype structures"""
        print("Loading wildtype data...")
        
        # Find wildtype files
        wt_cif_files = list(self.wildtype_dir.glob("*_model_*.cif"))
        wt_confidence_files = list(self.wildtype_dir.glob("*_summary_confidences_*.json"))
        
        if not wt_cif_files:
            print("Error: No wildtype CIF files found")
            return False
        
        # Process each wildtype model
        for i, cif_file in enumerate(sorted(wt_cif_files)[:3]):  # Use first 3 models
            print(f"Processing wildtype model {i}: {cif_file.name}")
            
            # Load confidence data
            confidence_data = {}
            confidence_dict = {}
            if i < len(wt_confidence_files):
                confidence_data, confidence_dict = self.load_confidence_data(wt_confidence_files[i])
            
            # Convert to PDB with confidence filtering
            pdb_file = self.wildtype_dir / f"wildtype_model_{i}_filtered.pdb"
            success = self.convert_cif_to_pdb(cif_file, pdb_file, confidence_dict)
            
            if success:
                self.wildtype_structures[i] = pdb_file
                self.wildtype_confidences[i] = {
                    'global_data': confidence_data,
                    'per_residue': confidence_dict,
                    'analysis': self.analyze_confidence_distribution(confidence_dict)
                }
        
        print(f"Loaded {len(self.wildtype_structures)} wildtype structures")
        return len(self.wildtype_structures) > 0
    
    def analyze_variant(self, variant_dir):
        """Analyze a single variant against wildtype"""
        variant_name = variant_dir.name
        print(f"Analyzing variant: {variant_name}")
        
        # Extract mutation info
        mutation = self.extract_mutation_info(variant_name)
        
        # Find variant files
        var_cif_files = list(variant_dir.glob("*_model_*.cif"))
        var_confidence_files = list(variant_dir.glob("*_summary_confidences_*.json"))
        
        if not var_cif_files:
            print(f"No CIF files found for {variant_name}")
            return None
        
        # Initialize results
        variant_results = {
            'variant_name': variant_name,
            'mutation': mutation,
            'num_models': len(var_cif_files)
        }
        
        all_comparisons = []
        
        # Process each variant model
        for i, var_cif in enumerate(sorted(var_cif_files)[:3]):  # Use first 3 models
            # Load confidence data
            var_confidence_data = {}
            var_confidence_dict = {}
            if i < len(var_confidence_files):
                var_confidence_data, var_confidence_dict = self.load_confidence_data(var_confidence_files[i])
            
            # Convert to PDB with confidence filtering
            var_pdb_file = variant_dir / f"variant_model_{i}_filtered.pdb"
            success = self.convert_cif_to_pdb(var_cif, var_pdb_file, var_confidence_dict)
            
            if not success:
                continue
            
            # Compare against each wildtype model
            for wt_idx, wt_pdb in self.wildtype_structures.items():
                print(f"  Comparing variant model {i} vs wildtype model {wt_idx}")
                
                # Try TM-align first
                tm_results = self.run_tmalign(wt_pdb, var_pdb_file)
                
                # Fallback to BioPython if TM-align fails
                if tm_results is None:
                    tm_results = self.calculate_rmsd_biopython(wt_pdb, var_pdb_file)
                
                if tm_results:
                    # Calculate GDT-TS
                    gdt_ts = self.calculate_gdt_ts(wt_pdb, var_pdb_file)
                    if gdt_ts is not None:
                        tm_results['gdt_ts'] = gdt_ts
                    
                    # Add confidence analysis
                    tm_results['variant_confidence'] = self.analyze_confidence_distribution(var_confidence_dict)
                    tm_results['wildtype_confidence'] = self.wildtype_confidences[wt_idx]['analysis']
                    
                    all_comparisons.append(tm_results)
        
        # Calculate aggregate metrics
        if all_comparisons:
            # RMSD statistics
            rmsds = [comp['rmsd'] for comp in all_comparisons if comp.get('rmsd')]
            if rmsds:
                variant_results['rmsd_mean'] = np.mean(rmsds)
                variant_results['rmsd_std'] = np.std(rmsds)
                variant_results['rmsd_min'] = np.min(rmsds)
                variant_results['rmsd_max'] = np.max(rmsds)
            
            # TM-score statistics (use best of both orientations)
            tm_scores = []
            for comp in all_comparisons:
                tm1 = comp.get('tm_score_1')
                tm2 = comp.get('tm_score_2')
                if tm1 and tm2:
                    tm_scores.append(max(tm1, tm2))
                elif tm1:
                    tm_scores.append(tm1)
                elif tm2:
                    tm_scores.append(tm2)
            
            if tm_scores:
                variant_results['tm_score_mean'] = np.mean(tm_scores)
                variant_results['tm_score_std'] = np.std(tm_scores)
                variant_results['tm_score_max'] = np.max(tm_scores)
                variant_results['tm_score_min'] = np.min(tm_scores)
            
            # GDT-TS statistics
            gdt_scores = [comp['gdt_ts'] for comp in all_comparisons if comp.get('gdt_ts')]
            if gdt_scores:
                variant_results['gdt_ts_mean'] = np.mean(gdt_scores)
                variant_results['gdt_ts_std'] = np.std(gdt_scores)
                variant_results['gdt_ts_max'] = np.max(gdt_scores)
                variant_results['gdt_ts_min'] = np.min(gdt_scores)
            
            # Alignment statistics
            aligned_lengths = [comp['aligned_length'] for comp in all_comparisons if comp.get('aligned_length')]
            if aligned_lengths:
                variant_results['aligned_length_mean'] = np.mean(aligned_lengths)
                variant_results['aligned_length_std'] = np.std(aligned_lengths)
            
            # Confidence statistics
            if all_comparisons[0].get('variant_confidence'):
                variant_results.update({
                    f"var_{k}": v for k, v in all_comparisons[0]['variant_confidence'].items()
                })
            
            # Add scientific interpretation
            self.add_scientific_interpretation(variant_results)
        
        return variant_results
    
    def extract_mutation_info(self, variant_name):
        """Extract mutation information from variant name"""
        parts = variant_name.split('_')
        if len(parts) >= 6:
            mutation = parts[-1]
            return mutation.upper()
        return variant_name
    
    def add_scientific_interpretation(self, results):
        """Add scientific interpretation based on metrics"""
        rmsd = results.get('rmsd_mean')
        tm_score = results.get('tm_score_mean')
        gdt_ts = results.get('gdt_ts_mean')
        
        interpretations = []
        
        if rmsd is not None:
            if rmsd < 1.0:
                interpretations.append("Nearly identical structures")
            elif rmsd < 2.0:
                interpretations.append("Very similar structures")
            elif rmsd < 4.0:
                interpretations.append("Similar overall fold")
            elif rmsd < 8.0:
                interpretations.append("Moderate structural differences")
            else:
                interpretations.append("Significant structural differences")
        
        if tm_score is not None:
            if tm_score > 0.8:
                interpretations.append("Excellent fold similarity")
            elif tm_score > 0.6:
                interpretations.append("Good fold similarity")
            elif tm_score > 0.4:
                interpretations.append("Moderate fold similarity")
            elif tm_score > 0.2:
                interpretations.append("Low fold similarity")
            else:
                interpretations.append("Different folds")
        
        if gdt_ts is not None:
            if gdt_ts > 80:
                interpretations.append("Excellent alignment quality")
            elif gdt_ts > 60:
                interpretations.append("Good alignment quality")
            elif gdt_ts > 40:
                interpretations.append("Moderate alignment quality")
            else:
                interpretations.append("Poor alignment quality")
        
        results['scientific_interpretation'] = "; ".join(interpretations)
        
        # Overall assessment
        if tm_score and tm_score > 0.5 and (rmsd is None or rmsd < 4.0):
            results['overall_assessment'] = "STRUCTURALLY_SIMILAR"
        elif tm_score and tm_score > 0.3:
            results['overall_assessment'] = "MODERATELY_DIFFERENT"
        else:
            results['overall_assessment'] = "SIGNIFICANTLY_DIFFERENT"
    
    def run_control_analysis(self):
        """Run wildtype vs wildtype control analysis"""
        print("\n" + "="*60)
        print("CONTROL ANALYSIS: Wildtype vs Wildtype Comparison")
        print("="*60)
        
        control_results = []
        wt_indices = list(self.wildtype_structures.keys())
        
        if len(wt_indices) < 2:
            print("Warning: Need at least 2 wildtype models for control analysis")
            return control_results
        
        # Compare each pair of wildtype models
        for i in range(len(wt_indices)):
            for j in range(i + 1, len(wt_indices)):
                wt_idx1, wt_idx2 = wt_indices[i], wt_indices[j]
                wt_pdb1 = self.wildtype_structures[wt_idx1]
                wt_pdb2 = self.wildtype_structures[wt_idx2]
                
                print(f"Control comparison: WT model {wt_idx1} vs WT model {wt_idx2}")
                
                # Run TM-align
                tm_results = self.run_tmalign(wt_pdb1, wt_pdb2)
                if tm_results is None:
                    tm_results = self.calculate_rmsd_biopython(wt_pdb1, wt_pdb2)
                
                if tm_results:
                    # Calculate GDT-TS
                    gdt_ts = self.calculate_gdt_ts(wt_pdb1, wt_pdb2)
                    if gdt_ts is not None:
                        tm_results['gdt_ts'] = gdt_ts
                    
                    control_result = {
                        'variant_name': f'WT_Control_{i}_{j}',
                        'mutation': f'WT_vs_WT_{wt_idx1}_{wt_idx2}',
                        'is_control': True,
                        'rmsd_mean': tm_results.get('rmsd'),
                        'tm_score_mean': max(tm_results.get('tm_score_1', 0), tm_results.get('tm_score_2', 0)),
                        'gdt_ts_mean': tm_results.get('gdt_ts'),
                        'aligned_length_mean': tm_results.get('aligned_length')
                    }
                    
                    self.add_scientific_interpretation(control_result)
                    control_results.append(control_result)
                    
                    print(f"  RMSD: {control_result['rmsd_mean']:.3f}Å | "
                          f"TM-Score: {control_result['tm_score_mean']:.3f} | "
                          f"GDT-TS: {control_result.get('gdt_ts_mean', 0):.1f}%")
        
        return control_results
    
    def run_analysis(self):
        """Run complete analysis"""
        print("Starting AlphaFold structural difference analysis...")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        if not BIOPYTHON_AVAILABLE:
            print("Error: BioPython is required. Install with: pip install biopython")
            return None
        
        # Load wildtype data
        if not self.load_wildtype_data():
            print("Error: Could not load wildtype data")
            return None
        
        # Run control analysis
        control_results = self.run_control_analysis()
        
        # Get variant directories
        variant_dirs = [d for d in self.base_dir.iterdir() 
                       if d.is_dir() and d.name != "p53_wildtype" and "p53_" in d.name]
        
        print(f"\nFound {len(variant_dirs)} variants to analyze")
        
        # Analyze each variant
        for variant_dir in sorted(variant_dirs):
            result = self.analyze_variant(variant_dir)
            if result:
                self.results.append(result)
        
        # Add control results
        self.results.extend(control_results)
        
        return self.results
    
    def generate_report(self, output_file="alphafold_structural_analysis.csv"):
        """Generate comprehensive report"""
        if not self.results:
            print("No results to report")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Separate control and variant data
        control_data = df[df.get('is_control', False) == True]
        variant_data = df[df.get('is_control', False) == False]
        
        # Save reports
        df.to_csv(output_file, index=False)
        print(f"Complete report saved to {output_file}")
        
        if not control_data.empty:
            control_data.to_csv("alphafold_control_analysis.csv", index=False)
            print("Control analysis saved to alphafold_control_analysis.csv")
        
        if not variant_data.empty:
            variant_data.to_csv("alphafold_variant_analysis.csv", index=False)
            print("Variant analysis saved to alphafold_variant_analysis.csv")
        
        # Print summary
        self.print_summary(control_data, variant_data)
        
        return df
    
    def print_summary(self, control_data, variant_data):
        """Print analysis summary"""
        print("\n" + "="*80)
        print("ALPHAFOLD STRUCTURAL ANALYSIS SUMMARY")
        print("="*80)
        
        # Control summary
        if not control_data.empty:
            print(f"\nCONTROL ANALYSIS (Wildtype vs Wildtype):")
            print("-" * 50)
            if 'rmsd_mean' in control_data.columns:
                rmsd_stats = control_data['rmsd_mean'].describe()
                print(f"RMSD: {rmsd_stats['mean']:.3f} ± {rmsd_stats['std']:.3f}Å "
                      f"(range: {rmsd_stats['min']:.3f} - {rmsd_stats['max']:.3f}Å)")
            
            if 'tm_score_mean' in control_data.columns:
                tm_stats = control_data['tm_score_mean'].describe()
                print(f"TM-Score: {tm_stats['mean']:.3f} ± {tm_stats['std']:.3f} "
                      f"(range: {tm_stats['min']:.3f} - {tm_stats['max']:.3f})")
            
            if 'gdt_ts_mean' in control_data.columns:
                gdt_stats = control_data['gdt_ts_mean'].describe()
                print(f"GDT-TS: {gdt_stats['mean']:.1f} ± {gdt_stats['std']:.1f}% "
                      f"(range: {gdt_stats['min']:.1f} - {gdt_stats['max']:.1f}%)")
        
        # Variant summary
        if not variant_data.empty:
            print(f"\nVARIANT ANALYSIS:")
            print("-" * 50)
            print(f"Total variants analyzed: {len(variant_data)}")
            
            if 'rmsd_mean' in variant_data.columns:
                best_variant = variant_data.loc[variant_data['rmsd_mean'].idxmin()]
                worst_variant = variant_data.loc[variant_data['rmsd_mean'].idxmax()]
                print(f"Most similar: {best_variant['mutation']} (RMSD: {best_variant['rmsd_mean']:.3f}Å)")
                print(f"Most different: {worst_variant['mutation']} (RMSD: {worst_variant['rmsd_mean']:.3f}Å)")
            
            # Assessment distribution
            if 'overall_assessment' in variant_data.columns:
                assessment_counts = variant_data['overall_assessment'].value_counts()
                print(f"\nOverall Assessment Distribution:")
                for assessment, count in assessment_counts.items():
                    print(f"  {assessment}: {count} variants ({count/len(variant_data)*100:.1f}%)")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='AlphaFold Structural Difference Analysis')
    parser.add_argument('base_directory', help='Base directory containing AlphaFold structures')
    parser.add_argument('--confidence-threshold', '-c', type=float, default=70.0,
                       help='Confidence threshold for filtering (default: 70.0)')
    parser.add_argument('--output', '-o', default='alphafold_structural_analysis.csv',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_directory):
        print(f"Error: Directory {args.base_directory} does not exist")
        return
    
    # Initialize analyzer
    analyzer = AlphaFoldStructuralAnalyzer(args.base_directory, args.confidence_threshold)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results:
        analyzer.generate_report(args.output)
    else:
        print("No results generated. Check your input data and dependencies.")


if __name__ == "__main__":
    main() 