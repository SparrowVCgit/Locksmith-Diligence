#!/usr/bin/env python3
"""
Protein Structure Similarity Analysis
Compares AlphaFold protein structures to wildtype p53 using multiple structural metrics.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.PDB.mmcifio import MMCIFIO
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio import BiopythonWarning
    warnings.simplefilter('ignore', BiopythonWarning)
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. Install with: pip install biopython")

try:
    import pymol
    from pymol import cmd
    PYMOL_AVAILABLE = True
except ImportError:
    PYMOL_AVAILABLE = False
    print("Warning: PyMOL not available. Some advanced metrics will be skipped.")

class StructuralSimilarityStandards:
    """Scientific standards for interpreting structural similarity metrics"""
    
    # RMSD interpretation (Angstroms) - for protein structures
    RMSD_THRESHOLDS = {
        'nearly_identical': 1.0,     # < 1.0 Å: Nearly identical structures
        'very_similar': 2.0,         # 1.0-2.0 Å: Very similar, minor local differences  
        'similar': 4.0,              # 2.0-4.0 Å: Similar overall fold, some significant differences
        'moderate_difference': 8.0,   # 4.0-8.0 Å: Different conformations but related fold
        # > 8.0 Å: Significantly different structures
    }
    
    # TM-Score interpretation (0-1 scale)
    TM_SCORE_THRESHOLDS = {
        'same_fold': 0.5,            # > 0.5: Same fold (high confidence)
        'mostly_same_fold': 0.4,     # 0.4-0.5: Mostly same fold
        'potentially_related': 0.2,   # 0.2-0.4: Potentially related folds
        # < 0.2: Different folds
    }
    
    # GDT-TS interpretation (percentage)
    GDT_TS_THRESHOLDS = {
        'high_similarity': 50.0,     # > 50%: High similarity
        'moderate_similarity': 30.0,  # 30-50%: Moderate similarity  
        'low_similarity': 10.0,      # 10-30%: Low similarity
        # < 10%: Very low similarity
    }
    
    @staticmethod
    def interpret_rmsd(rmsd_value):
        """Interpret RMSD value according to scientific standards"""
        if rmsd_value < StructuralSimilarityStandards.RMSD_THRESHOLDS['nearly_identical']:
            return "Nearly Identical", "Very high structural similarity, essentially the same structure"
        elif rmsd_value < StructuralSimilarityStandards.RMSD_THRESHOLDS['very_similar']:
            return "Very Similar", "High structural similarity with minor local differences"
        elif rmsd_value < StructuralSimilarityStandards.RMSD_THRESHOLDS['similar']:
            return "Similar", "Good structural similarity, same overall fold with some differences"
        elif rmsd_value < StructuralSimilarityStandards.RMSD_THRESHOLDS['moderate_difference']:
            return "Moderate Difference", "Different conformations but likely related fold"
        else:
            return "Significantly Different", "Major structural differences, possibly different folds"
    
    @staticmethod
    def interpret_tm_score(tm_score):
        """Interpret TM-Score according to scientific standards"""
        if tm_score > StructuralSimilarityStandards.TM_SCORE_THRESHOLDS['same_fold']:
            return "Same Fold", "High confidence that structures have the same fold"
        elif tm_score > StructuralSimilarityStandards.TM_SCORE_THRESHOLDS['mostly_same_fold']:
            return "Mostly Same Fold", "Good confidence of similar fold with some differences"
        elif tm_score > StructuralSimilarityStandards.TM_SCORE_THRESHOLDS['potentially_related']:
            return "Potentially Related", "Possibly related folds, requires careful examination"
        else:
            return "Different Folds", "Likely different protein folds"
    
    @staticmethod
    def interpret_gdt_ts(gdt_ts):
        """Interpret GDT-TS score according to scientific standards"""
        if gdt_ts > StructuralSimilarityStandards.GDT_TS_THRESHOLDS['high_similarity']:
            return "High Similarity", "Excellent structural alignment quality"
        elif gdt_ts > StructuralSimilarityStandards.GDT_TS_THRESHOLDS['moderate_similarity']:
            return "Moderate Similarity", "Good structural alignment with some differences"
        elif gdt_ts > StructuralSimilarityStandards.GDT_TS_THRESHOLDS['low_similarity']:
            return "Low Similarity", "Limited structural similarity"
        else:
            return "Very Low Similarity", "Poor structural alignment, likely different structures"
    
    @staticmethod
    def get_biological_significance(rmsd_value, tm_score, gdt_ts):
        """Determine overall biological significance of structural differences"""
        rmsd_interp, _ = StructuralSimilarityStandards.interpret_rmsd(rmsd_value)
        tm_interp, _ = StructuralSimilarityStandards.interpret_tm_score(tm_score)
        gdt_interp, _ = StructuralSimilarityStandards.interpret_gdt_ts(gdt_ts)
        
        # Conservative interpretation - if any metric suggests significant difference
        if (rmsd_value > StructuralSimilarityStandards.RMSD_THRESHOLDS['moderate_difference'] or
            tm_score < StructuralSimilarityStandards.TM_SCORE_THRESHOLDS['potentially_related'] or
            gdt_ts < StructuralSimilarityStandards.GDT_TS_THRESHOLDS['low_similarity']):
            return "BIOLOGICALLY SIGNIFICANT", "Major structural changes likely to affect function"
        elif (rmsd_value > StructuralSimilarityStandards.RMSD_THRESHOLDS['similar'] or
              tm_score < StructuralSimilarityStandards.TM_SCORE_THRESHOLDS['mostly_same_fold'] or
              gdt_ts < StructuralSimilarityStandards.GDT_TS_THRESHOLDS['moderate_similarity']):
            return "POTENTIALLY SIGNIFICANT", "Moderate changes that may affect function"
        else:
            return "LIKELY NOT SIGNIFICANT", "Minor structural changes unlikely to majorly affect function"

class ProteinStructureAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.wildtype_dir = self.base_dir / "p53_wildtype"
        self.results = []
        
    def get_variant_directories(self):
        """Get all variant directories (excluding wildtype)"""
        variant_dirs = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name != "p53_wildtype" and "p53_" in item.name:
                variant_dirs.append(item)
        return sorted(variant_dirs)
    
    def extract_mutation_info(self, variant_name):
        """Extract mutation information from variant name"""
        # Expected format: 2025_06_01_14_51_p53_r175p
        parts = variant_name.split('_')
        if len(parts) >= 6:
            mutation = parts[-1]  # e.g., r175p
            return mutation.upper()
        return variant_name
    
    def load_cif_structure(self, cif_file):
        """Load structure from CIF file using BioPython"""
        if not BIOPYTHON_AVAILABLE:
            return None
        
        parser = MMCIFParser(QUIET=True)
        try:
            structure = parser.get_structure("protein", cif_file)
            return structure
        except Exception as e:
            print(f"Error loading {cif_file}: {e}")
            return None
    
    def load_confidence_data(self, json_file):
        """Load confidence data from JSON file"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return None
    
    def calculate_rmsd(self, structure1, structure2):
        """Calculate RMSD between two structures using CA atoms"""
        if not BIOPYTHON_AVAILABLE or structure1 is None or structure2 is None:
            return None
        
        try:
            from Bio.PDB import Superimposer
            
            # Get CA atoms from both structures
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
            
            # Ensure same number of atoms
            min_len = min(len(ca_atoms_1), len(ca_atoms_2))
            if min_len == 0:
                return None
            
            ca_atoms_1 = ca_atoms_1[:min_len]
            ca_atoms_2 = ca_atoms_2[:min_len]
            
            # Calculate RMSD
            super_imposer = Superimposer()
            super_imposer.set_atoms(ca_atoms_1, ca_atoms_2)
            
            return super_imposer.rms
            
        except Exception as e:
            print(f"Error calculating RMSD: {e}")
            return None
    
    def calculate_gdt_ts(self, structure1, structure2, thresholds=[1, 2, 4, 8]):
        """Calculate GDT-TS score"""
        if not BIOPYTHON_AVAILABLE or structure1 is None or structure2 is None:
            return None
        
        try:
            from Bio.PDB import Superimposer
            
            # Get CA atom coordinates
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
            if min_len == 0:
                return None
            
            coords1 = np.array(coords1[:min_len])
            coords2 = np.array(coords2[:min_len])
            
            # Align structures
            super_imposer = Superimposer()
            # Create dummy atom objects for superimposition
            from Bio.PDB.Atom import Atom
            atoms1 = [Atom("CA", coord, 0, 1, " ", "CA", i, "C") for i, coord in enumerate(coords1)]
            atoms2 = [Atom("CA", coord, 0, 1, " ", "CA", i, "C") for i, coord in enumerate(coords2)]
            
            super_imposer.set_atoms(atoms1, atoms2)
            super_imposer.apply(atoms2)
            
            # Calculate distances after alignment
            aligned_coords2 = np.array([atom.coord for atom in atoms2])
            distances = np.linalg.norm(coords1 - aligned_coords2, axis=1)
            
            # Calculate GDT-TS
            gdt_scores = []
            for threshold in thresholds:
                within_threshold = np.sum(distances <= threshold)
                gdt_scores.append(within_threshold / len(distances))
            
            gdt_ts = np.mean(gdt_scores) * 100
            return gdt_ts
            
        except Exception as e:
            print(f"Error calculating GDT-TS: {e}")
            return None
    
    def calculate_tm_score_simple(self, structure1, structure2):
        """Simple TM-Score approximation"""
        if not BIOPYTHON_AVAILABLE or structure1 is None or structure2 is None:
            return None
        
        try:
            from Bio.PDB import Superimposer
            
            # Get CA atom coordinates
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
            if min_len == 0:
                return None
            
            L = len(coords1)  # Length of target structure
            d0 = 1.24 * (L - 15)**(1/3) - 1.8  # TM-Score normalization factor
            
            coords1 = np.array(coords1[:min_len])
            coords2 = np.array(coords2[:min_len])
            
            # Align structures
            from Bio.PDB.Atom import Atom
            atoms1 = [Atom("CA", coord, 0, 1, " ", "CA", i, "C") for i, coord in enumerate(coords1)]
            atoms2 = [Atom("CA", coord, 0, 1, " ", "CA", i, "C") for i, coord in enumerate(coords2)]
            
            super_imposer = Superimposer()
            super_imposer.set_atoms(atoms1, atoms2)
            super_imposer.apply(atoms2)
            
            # Calculate TM-Score
            aligned_coords2 = np.array([atom.coord for atom in atoms2])
            distances = np.linalg.norm(coords1 - aligned_coords2, axis=1)
            
            tm_score = np.sum(1 / (1 + (distances / d0)**2)) / L
            return tm_score
            
        except Exception as e:
            print(f"Error calculating TM-Score: {e}")
            return None
    
    def calculate_sequence_identity(self, structure1, structure2):
        """Calculate sequence identity between structures"""
        if not BIOPYTHON_AVAILABLE or structure1 is None or structure2 is None:
            return None
        
        try:
            # Extract sequences
            seq1, seq2 = [], []
            
            for model in structure1:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':  # Standard amino acid
                            seq1.append(residue.resname)
            
            for model in structure2:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':  # Standard amino acid
                            seq2.append(residue.resname)
            
            min_len = min(len(seq1), len(seq2))
            if min_len == 0:
                return None
            
            identical = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
            return (identical / min_len) * 100
            
        except Exception as e:
            print(f"Error calculating sequence identity: {e}")
            return None
    
    def calculate_confidence_metrics(self, wt_confidence, var_confidence):
        """Calculate confidence-based metrics"""
        if not wt_confidence or not var_confidence:
            return {}
        
        metrics = {}
        
        # Compare confidence scores
        if 'ptm' in wt_confidence and 'ptm' in var_confidence:
            metrics['ptm_difference'] = abs(wt_confidence['ptm'] - var_confidence['ptm'])
            metrics['ptm_ratio'] = var_confidence['ptm'] / wt_confidence['ptm'] if wt_confidence['ptm'] > 0 else None
        
        if 'ranking_score' in wt_confidence and 'ranking_score' in var_confidence:
            metrics['ranking_score_difference'] = abs(wt_confidence['ranking_score'] - var_confidence['ranking_score'])
            metrics['ranking_score_ratio'] = var_confidence['ranking_score'] / wt_confidence['ranking_score'] if wt_confidence['ranking_score'] > 0 else None
        
        if 'fraction_disordered' in wt_confidence and 'fraction_disordered' in var_confidence:
            metrics['disorder_difference'] = abs(wt_confidence['fraction_disordered'] - var_confidence['fraction_disordered'])
        
        return metrics
    
    def analyze_variant(self, variant_dir, wildtype_structures, wildtype_confidences):
        """Analyze a single variant against wildtype"""
        variant_name = variant_dir.name
        mutation = self.extract_mutation_info(variant_name)
        
        print(f"Analyzing variant: {mutation}")
        
        # Find CIF files for this variant
        cif_files = list(variant_dir.glob("*_model_*.cif"))
        confidence_files = list(variant_dir.glob("*_summary_confidences_*.json"))
        
        if not cif_files:
            print(f"No CIF files found for {variant_name}")
            return None
        
        variant_results = {
            'variant_name': variant_name,
            'mutation': mutation,
            'num_models': len(cif_files)
        }
        
        # Analyze each model against wildtype models
        all_rmsd = []
        all_gdt_ts = []
        all_tm_scores = []
        all_seq_identity = []
        
        for i, var_cif in enumerate(sorted(cif_files)[:3]):  # Limit to first 3 models
            var_structure = self.load_cif_structure(var_cif)
            
            if var_structure is None:
                continue
            
            # Compare against wildtype models
            for j, (wt_structure, _) in enumerate(wildtype_structures[:3]):  # First 3 WT models
                if wt_structure is None:
                    continue
                
                # Calculate structural metrics
                rmsd = self.calculate_rmsd(wt_structure, var_structure)
                gdt_ts = self.calculate_gdt_ts(wt_structure, var_structure)
                tm_score = self.calculate_tm_score_simple(wt_structure, var_structure)
                seq_identity = self.calculate_sequence_identity(wt_structure, var_structure)
                
                if rmsd is not None:
                    all_rmsd.append(rmsd)
                if gdt_ts is not None:
                    all_gdt_ts.append(gdt_ts)
                if tm_score is not None:
                    all_tm_scores.append(tm_score)
                if seq_identity is not None:
                    all_seq_identity.append(seq_identity)
        
        # Calculate average metrics
        if all_rmsd:
            variant_results['rmsd_mean'] = np.mean(all_rmsd)
            variant_results['rmsd_std'] = np.std(all_rmsd)
            variant_results['rmsd_min'] = np.min(all_rmsd)
        
        if all_gdt_ts:
            variant_results['gdt_ts_mean'] = np.mean(all_gdt_ts)
            variant_results['gdt_ts_std'] = np.std(all_gdt_ts)
            variant_results['gdt_ts_max'] = np.max(all_gdt_ts)
        
        if all_tm_scores:
            variant_results['tm_score_mean'] = np.mean(all_tm_scores)
            variant_results['tm_score_std'] = np.std(all_tm_scores)
            variant_results['tm_score_max'] = np.max(all_tm_scores)
        
        if all_seq_identity:
            variant_results['sequence_identity'] = np.mean(all_seq_identity)
        
        # Add scientific interpretations
        if all_rmsd and all_gdt_ts and all_tm_scores:
            rmsd_interp, rmsd_desc = StructuralSimilarityStandards.interpret_rmsd(variant_results['rmsd_mean'])
            tm_interp, tm_desc = StructuralSimilarityStandards.interpret_tm_score(variant_results['tm_score_mean'])
            gdt_interp, gdt_desc = StructuralSimilarityStandards.interpret_gdt_ts(variant_results['gdt_ts_mean'])
            bio_sig, bio_desc = StructuralSimilarityStandards.get_biological_significance(
                variant_results['rmsd_mean'], variant_results['tm_score_mean'], variant_results['gdt_ts_mean'])
            
            variant_results['rmsd_interpretation'] = rmsd_interp
            variant_results['rmsd_description'] = rmsd_desc
            variant_results['tm_score_interpretation'] = tm_interp
            variant_results['tm_score_description'] = tm_desc
            variant_results['gdt_ts_interpretation'] = gdt_interp
            variant_results['gdt_ts_description'] = gdt_desc
            variant_results['biological_significance'] = bio_sig
            variant_results['biological_description'] = bio_desc
        
        # Analyze confidence data
        if confidence_files and wildtype_confidences:
            var_confidence = self.load_confidence_data(confidence_files[0])
            confidence_metrics = self.calculate_confidence_metrics(wildtype_confidences[0], var_confidence)
            variant_results.update(confidence_metrics)
            
            # Add individual confidence scores
            if var_confidence:
                variant_results['variant_ptm'] = var_confidence.get('ptm')
                variant_results['variant_ranking_score'] = var_confidence.get('ranking_score')
                variant_results['variant_fraction_disordered'] = var_confidence.get('fraction_disordered')
        
        return variant_results
    
    def load_wildtype_data(self):
        """Load wildtype structure and confidence data"""
        print("Loading wildtype data...")
        
        # Load wildtype structures
        wt_cif_files = list(self.wildtype_dir.glob("*_model_*.cif"))
        wt_confidence_files = list(self.wildtype_dir.glob("*_summary_confidences_*.json"))
        
        wildtype_structures = []
        for cif_file in sorted(wt_cif_files):
            structure = self.load_cif_structure(cif_file)
            wildtype_structures.append((structure, cif_file.name))
        
        wildtype_confidences = []
        for conf_file in sorted(wt_confidence_files):
            confidence = self.load_confidence_data(conf_file)
            wildtype_confidences.append(confidence)
        
        return wildtype_structures, wildtype_confidences
    
    def run_analysis(self):
        """Run complete analysis"""
        print("Starting protein structure similarity analysis...")
        
        if not BIOPYTHON_AVAILABLE:
            print("Error: BioPython is required for structural analysis.")
            print("Install with: pip install biopython")
            return None
        
        # Load wildtype data
        wildtype_structures, wildtype_confidences = self.load_wildtype_data()
        
        if not wildtype_structures:
            print("Error: Could not load wildtype structures")
            return None
        
        # CONTROL ANALYSIS: Compare wildtype models against each other
        print("\n" + "="*60)
        print("CONTROL ANALYSIS: Wildtype vs Wildtype Comparison")
        print("="*60)
        print("This establishes baseline measurement error and validates thresholds...")
        
        wt_control_results = self.analyze_wildtype_control(wildtype_structures, wildtype_confidences)
        
        # Get variant directories
        variant_dirs = self.get_variant_directories()
        print(f"\nFound {len(variant_dirs)} variants to analyze")
        
        # Analyze each variant
        for variant_dir in variant_dirs:
            result = self.analyze_variant(variant_dir, wildtype_structures, wildtype_confidences)
            if result:
                self.results.append(result)
        
        # Add control results to the main results for comparison
        if wt_control_results:
            self.results.extend(wt_control_results)
        
        return self.results
    
    def analyze_wildtype_control(self, wildtype_structures, wildtype_confidences):
        """Analyze wildtype models against each other as control"""
        control_results = []
        
        if len(wildtype_structures) < 2:
            print("Warning: Need at least 2 wildtype models for control analysis")
            return control_results
        
        # Compare each wildtype model against the first one (reference)
        reference_structure, ref_name = wildtype_structures[0]
        
        for i, (wt_structure, wt_name) in enumerate(wildtype_structures[1:], 1):
            if reference_structure is None or wt_structure is None:
                continue
            
            print(f"Control comparison {i}: {ref_name} vs {wt_name}")
            
            # Calculate all metrics between wildtype models
            rmsd = self.calculate_rmsd(reference_structure, wt_structure)
            gdt_ts = self.calculate_gdt_ts(reference_structure, wt_structure)
            tm_score = self.calculate_tm_score_simple(reference_structure, wt_structure)
            seq_identity = self.calculate_sequence_identity(reference_structure, wt_structure)
            
            control_result = {
                'variant_name': f'WT_Control_{i}',
                'mutation': f'WT_vs_WT_{i}',
                'num_models': 1,
                'is_control': True,  # Flag to identify control samples
                'control_type': 'wildtype_vs_wildtype'
            }
            
            if rmsd is not None:
                control_result['rmsd_mean'] = rmsd
                control_result['rmsd_std'] = 0.0  # Single comparison
                control_result['rmsd_min'] = rmsd
            
            if gdt_ts is not None:
                control_result['gdt_ts_mean'] = gdt_ts
                control_result['gdt_ts_std'] = 0.0
                control_result['gdt_ts_max'] = gdt_ts
            
            if tm_score is not None:
                control_result['tm_score_mean'] = tm_score
                control_result['tm_score_std'] = 0.0
                control_result['tm_score_max'] = tm_score
            
            if seq_identity is not None:
                control_result['sequence_identity'] = seq_identity
            
            # Add scientific interpretations for control
            if rmsd is not None and gdt_ts is not None and tm_score is not None:
                rmsd_interp, rmsd_desc = StructuralSimilarityStandards.interpret_rmsd(rmsd)
                tm_interp, tm_desc = StructuralSimilarityStandards.interpret_tm_score(tm_score)
                gdt_interp, gdt_desc = StructuralSimilarityStandards.interpret_gdt_ts(gdt_ts)
                bio_sig, bio_desc = StructuralSimilarityStandards.get_biological_significance(rmsd, tm_score, gdt_ts)
                
                control_result['rmsd_interpretation'] = rmsd_interp
                control_result['rmsd_description'] = rmsd_desc
                control_result['tm_score_interpretation'] = tm_interp
                control_result['tm_score_description'] = tm_desc
                control_result['gdt_ts_interpretation'] = gdt_interp
                control_result['gdt_ts_description'] = gdt_desc
                control_result['biological_significance'] = bio_sig
                control_result['biological_description'] = bio_desc
            
            # Add confidence comparison if available
            if wildtype_confidences and len(wildtype_confidences) > i:
                ref_confidence = wildtype_confidences[0]
                wt_confidence = wildtype_confidences[i]
                confidence_metrics = self.calculate_confidence_metrics(ref_confidence, wt_confidence)
                control_result.update(confidence_metrics)
                
                if wt_confidence:
                    control_result['variant_ptm'] = wt_confidence.get('ptm')
                    control_result['variant_ranking_score'] = wt_confidence.get('ranking_score')
                    control_result['variant_fraction_disordered'] = wt_confidence.get('fraction_disordered')
            
            control_results.append(control_result)
            
            # Print control results
            print(f"  RMSD: {rmsd:.3f}Å | GDT-TS: {gdt_ts:.1f}% | TM-Score: {tm_score:.3f} | {bio_sig}")
        
        # Calculate control statistics
        if control_results:
            control_rmsds = [r['rmsd_mean'] for r in control_results if 'rmsd_mean' in r]
            control_gdt_ts = [r['gdt_ts_mean'] for r in control_results if 'gdt_ts_mean' in r]
            control_tm_scores = [r['tm_score_mean'] for r in control_results if 'tm_score_mean' in r]
            
            print(f"\nCONTROL STATISTICS (Wildtype vs Wildtype):")
            print("-" * 50)
            if control_rmsds:
                print(f"RMSD Range: {min(control_rmsds):.3f} - {max(control_rmsds):.3f}Å")
                print(f"RMSD Mean ± SD: {np.mean(control_rmsds):.3f} ± {np.std(control_rmsds):.3f}Å")
            if control_gdt_ts:
                print(f"GDT-TS Range: {min(control_gdt_ts):.1f} - {max(control_gdt_ts):.1f}%")
                print(f"GDT-TS Mean ± SD: {np.mean(control_gdt_ts):.1f} ± {np.std(control_gdt_ts):.1f}%")
            if control_tm_scores:
                print(f"TM-Score Range: {min(control_tm_scores):.3f} - {max(control_tm_scores):.3f}")
                print(f"TM-Score Mean ± SD: {np.mean(control_tm_scores):.3f} ± {np.std(control_tm_scores):.3f}")
            
            print(f"\nINTERPRETATION:")
            print("These control values represent the baseline measurement error.")
            print("Variants with differences significantly larger than these controls")
            print("can be considered to have meaningful structural changes.")
            print("="*60)
        
        return control_results
    
    def generate_report(self, output_file="protein_similarity_report.csv"):
        """Generate comprehensive report"""
        if not self.results:
            print("No results to report")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Separate control and variant data
        control_data = df[df.get('is_control', False) == True] if 'is_control' in df.columns else pd.DataFrame()
        variant_data = df[df.get('is_control', False) == False] if 'is_control' in df.columns else df
        
        # Sort variants by structural similarity (lower RMSD = more similar)
        if 'rmsd_mean' in variant_data.columns:
            variant_data = variant_data.sort_values('rmsd_mean')
        
        # Save full data to CSV
        df.to_csv(output_file, index=False)
        print(f"Report saved to {output_file}")
        
        # Save separate control and variant reports
        if not control_data.empty:
            control_data.to_csv("control_analysis_report.csv", index=False)
            print(f"Control analysis saved to control_analysis_report.csv")
        
        if not variant_data.empty:
            variant_data.to_csv("variant_analysis_report.csv", index=False)
            print(f"Variant analysis saved to variant_analysis_report.csv")
        
        # Print summary statistics with scientific interpretation
        print("\n" + "="*80)
        print("PROTEIN STRUCTURE SIMILARITY ANALYSIS REPORT")
        print("="*80)
        
        # CONTROL ANALYSIS SUMMARY
        if not control_data.empty:
            print("\nCONTROL ANALYSIS SUMMARY (Wildtype vs Wildtype):")
            print("-" * 60)
            print("This establishes the baseline measurement error for our methods.")
            
            if 'rmsd_mean' in control_data.columns:
                control_rmsd_mean = control_data['rmsd_mean'].mean()
                control_rmsd_std = control_data['rmsd_mean'].std()
                control_rmsd_max = control_data['rmsd_mean'].max()
                print(f"Control RMSD: {control_rmsd_mean:.3f} ± {control_rmsd_std:.3f}Å (max: {control_rmsd_max:.3f}Å)")
            
            if 'gdt_ts_mean' in control_data.columns:
                control_gdt_mean = control_data['gdt_ts_mean'].mean()
                control_gdt_std = control_data['gdt_ts_mean'].std()
                control_gdt_min = control_data['gdt_ts_mean'].min()
                print(f"Control GDT-TS: {control_gdt_mean:.1f} ± {control_gdt_std:.1f}% (min: {control_gdt_min:.1f}%)")
            
            if 'tm_score_mean' in control_data.columns:
                control_tm_mean = control_data['tm_score_mean'].mean()
                control_tm_std = control_data['tm_score_mean'].std()
                control_tm_min = control_data['tm_score_mean'].min()
                print(f"Control TM-Score: {control_tm_mean:.3f} ± {control_tm_std:.3f} (min: {control_tm_min:.3f})")
            
            print(f"\n⚠️  IMPORTANT: Any variant with RMSD > {control_rmsd_max:.3f}Å shows")
            print(f"   differences beyond measurement error and likely represents real structural changes.")
        
        print("\nSCIENTIFIC INTERPRETATION GUIDE:")
        print("-" * 40)
        print("RMSD Thresholds (Å):")
        print("  < 1.0 : Nearly identical structures")
        print("  1.0-2.0 : Very similar, minor differences")
        print("  2.0-4.0 : Similar fold, some differences")
        print("  4.0-8.0 : Different conformations, related fold")
        print("  > 8.0 : Significantly different structures")
        print("\nTM-Score Thresholds:")
        print("  > 0.5 : Same fold (high confidence)")
        print("  0.4-0.5 : Mostly same fold")
        print("  0.2-0.4 : Potentially related folds")
        print("  < 0.2 : Different folds")
        print("\nGDT-TS Thresholds (%):")
        print("  > 50 : High similarity")
        print("  30-50 : Moderate similarity")
        print("  10-30 : Low similarity")
        print("  < 10 : Very low similarity")
        
        # VARIANT ANALYSIS
        if not variant_data.empty:
            print(f"\nVARIANT ANALYSIS SUMMARY:")
            print("-" * 40)
            print(f"Total variants analyzed: {len(variant_data)}")
            
            if 'rmsd_mean' in variant_data.columns:
                print(f"\nRMSD Statistics (Å):")
                print(f"  Best similarity (lowest RMSD): {variant_data['rmsd_mean'].min():.3f} Å ({variant_data.loc[variant_data['rmsd_mean'].idxmin(), 'mutation']})")
                print(f"  Worst similarity (highest RMSD): {variant_data['rmsd_mean'].max():.3f} Å ({variant_data.loc[variant_data['rmsd_mean'].idxmax(), 'mutation']})")
                print(f"  Average RMSD: {variant_data['rmsd_mean'].mean():.3f} ± {variant_data['rmsd_mean'].std():.3f} Å")
                
                # Compare to control baseline
                if not control_data.empty and 'rmsd_mean' in control_data.columns:
                    control_threshold = control_data['rmsd_mean'].max()
                    significant_variants = variant_data[variant_data['rmsd_mean'] > control_threshold]
                    print(f"  Variants beyond control baseline (RMSD > {control_threshold:.3f}Å): {len(significant_variants)} ({len(significant_variants)/len(variant_data)*100:.1f}%)")
            
            if 'gdt_ts_mean' in variant_data.columns:
                print(f"\nGDT-TS Statistics (%):")
                print(f"  Best similarity (highest GDT-TS): {variant_data['gdt_ts_mean'].max():.1f}% ({variant_data.loc[variant_data['gdt_ts_mean'].idxmax(), 'mutation']})")
                print(f"  Worst similarity (lowest GDT-TS): {variant_data['gdt_ts_mean'].min():.1f}% ({variant_data.loc[variant_data['gdt_ts_mean'].idxmin(), 'mutation']})")
                print(f"  Average GDT-TS: {variant_data['gdt_ts_mean'].mean():.1f} ± {variant_data['gdt_ts_mean'].std():.1f}%")
            
            if 'tm_score_mean' in variant_data.columns:
                print(f"\nTM-Score Statistics:")
                print(f"  Best similarity (highest TM-Score): {variant_data['tm_score_mean'].max():.3f} ({variant_data.loc[variant_data['tm_score_mean'].idxmax(), 'mutation']})")
                print(f"  Worst similarity (lowest TM-Score): {variant_data['tm_score_mean'].min():.3f} ({variant_data.loc[variant_data['tm_score_mean'].idxmin(), 'mutation']})")
                print(f"  Average TM-Score: {variant_data['tm_score_mean'].mean():.3f} ± {variant_data['tm_score_mean'].std():.3f}")
            
            if 'sequence_identity' in variant_data.columns:
                print(f"\nSequence Identity Statistics (%):")
                print(f"  Highest identity: {variant_data['sequence_identity'].max():.1f}% ({variant_data.loc[variant_data['sequence_identity'].idxmax(), 'mutation']})")
                print(f"  Lowest identity: {variant_data['sequence_identity'].min():.1f}% ({variant_data.loc[variant_data['sequence_identity'].idxmin(), 'mutation']})")
                print(f"  Average identity: {variant_data['sequence_identity'].mean():.1f} ± {variant_data['sequence_identity'].std():.1f}%")
            
            # Biological significance summary
            if 'biological_significance' in variant_data.columns:
                print(f"\nBIOLOGICAL SIGNIFICANCE SUMMARY:")
                print("-" * 40)
                bio_counts = variant_data['biological_significance'].value_counts()
                for significance, count in bio_counts.items():
                    print(f"  {significance}: {count} variants ({count/len(variant_data)*100:.1f}%)")
            
            print(f"\nTop 5 Most Similar Variants (by RMSD):")
            print("-" * 70)
            if 'rmsd_mean' in variant_data.columns and 'biological_significance' in variant_data.columns:
                top5 = variant_data.nsmallest(5, 'rmsd_mean')[['mutation', 'rmsd_mean', 'gdt_ts_mean', 'tm_score_mean', 'biological_significance']]
                for _, row in top5.iterrows():
                    print(f"  {row['mutation']:10s}: RMSD={row['rmsd_mean']:.3f}Å, GDT-TS={row['gdt_ts_mean']:.1f}%, TM-Score={row['tm_score_mean']:.3f} | {row['biological_significance']}")
            
            print(f"\nTop 5 Most Different Variants (by RMSD):")
            print("-" * 70)
            if 'rmsd_mean' in variant_data.columns and 'biological_significance' in variant_data.columns:
                bottom5 = variant_data.nlargest(5, 'rmsd_mean')[['mutation', 'rmsd_mean', 'gdt_ts_mean', 'tm_score_mean', 'biological_significance']]
                for _, row in bottom5.iterrows():
                    print(f"  {row['mutation']:10s}: RMSD={row['rmsd_mean']:.3f}Å, GDT-TS={row['gdt_ts_mean']:.1f}%, TM-Score={row['tm_score_mean']:.3f} | {row['biological_significance']}")
        
        print("\n" + "="*80)
        print("Analysis complete!")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Analyze protein structure similarity')
    parser.add_argument('base_directory', help='Base directory containing protein structures')
    parser.add_argument('--output', '-o', default='protein_similarity_report.csv', 
                       help='Output file for results (default: protein_similarity_report.csv)')
    
    args = parser.parse_args()
    
    # Check if base directory exists
    if not os.path.exists(args.base_directory):
        print(f"Error: Directory {args.base_directory} does not exist")
        return
    
    # Initialize analyzer
    analyzer = ProteinStructureAnalyzer(args.base_directory)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results:
        # Generate report
        analyzer.generate_report(args.output)
    else:
        print("No results generated. Check if BioPython is installed and structures are valid.")


if __name__ == "__main__":
    main() 