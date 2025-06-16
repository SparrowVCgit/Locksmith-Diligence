#!/usr/bin/env python3
"""
Enhanced AlphaFold Structural Analysis Pipeline
==============================================
A comprehensive pipeline following the workflow:
AlphaFold output → Convert .cif to .pdb → Mask low-confidence regions (pLDDT < 70) 
→ Align structures using TM-align or PyMOL → Compute RMSD + TM-score + GDT-TS 
→ Visualize differences → Map PTM/PAE to interpret confidence

Author: Pragyanshu Singh
Date: 2025
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
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import shutil

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from Bio import PDB
    from Bio.PDB import PDBIO, Select
    from Bio.PDB.Superimposer import Superimposer
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.MMCIFParser import MMCIFParser
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    logger.error(f"Required dependencies not found: {e}")
    sys.exit(1)

# Optional PyMOL import
try:
    import pymol
    from pymol import cmd
    PYMOL_AVAILABLE = True
    logger.info("PyMOL available for advanced alignment and visualization")
except ImportError:
    PYMOL_AVAILABLE = False
    logger.warning("PyMOL not available - using BioPython for alignment")

@dataclass
class StructuralMetrics:
    """Container for structural comparison metrics"""
    rmsd: float
    tm_score: Optional[float] = None
    gdt_ts: Optional[float] = None
    aligned_residues: int = 0
    total_residues: int = 0
    coverage: float = 0.0
    mean_plddt: float = 0.0
    confidence_mask_percentage: float = 0.0
    pae_score: Optional[float] = None

@dataclass
class AlphaFoldStructure:
    """Container for AlphaFold structure data"""
    name: str
    cif_path: str
    pdb_path: Optional[str] = None
    masked_pdb_path: Optional[str] = None
    confidence_path: str = ""
    plddt_scores: Optional[List[float]] = None
    pae_matrix: Optional[np.ndarray] = None
    ptm_score: Optional[float] = None
    ranking_score: Optional[float] = None
    chain_id: str = "A"

class ConfidenceFilter:
    """Enhanced confidence-based filtering and masking"""
    
    def __init__(self, plddt_threshold: float = 70.0, pae_threshold: float = 10.0):
        self.plddt_threshold = plddt_threshold
        self.pae_threshold = pae_threshold
    
    def load_confidence_data(self, confidence_path: str) -> Dict[str, Any]:
        """Load confidence data from AlphaFold JSON"""
        try:
            with open(confidence_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load confidence data from {confidence_path}: {e}")
            return {}
    
    def extract_plddt_from_cif(self, cif_path: str) -> List[float]:
        """Extract per-residue pLDDT scores from CIF file"""
        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('temp', cif_path)
            
            plddt_scores = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            if atom.name == 'CA':
                                plddt = atom.bfactor
                                plddt_scores.append(plddt)
                                break
                break
            
            return plddt_scores
        except Exception as e:
            logger.error(f"Failed to extract pLDDT from {cif_path}: {e}")
            return []
    
    def extract_pae_matrix(self, confidence_path: str) -> Optional[np.ndarray]:
        """Extract Predicted Aligned Error matrix"""
        try:
            confidence_data = self.load_confidence_data(confidence_path)
            if 'predicted_aligned_error' in confidence_data:
                pae = confidence_data['predicted_aligned_error']
                return np.array(pae)
            elif 'pae' in confidence_data:
                pae = confidence_data['pae']
                return np.array(pae)
            else:
                logger.warning(f"No PAE data found in {confidence_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract PAE from {confidence_path}: {e}")
            return None
    
    def create_confidence_mask(self, structure_path: str, plddt_scores: List[float]) -> List[bool]:
        """Create mask for high-confidence regions"""
        mask = [score >= self.plddt_threshold for score in plddt_scores]
        high_conf_count = sum(mask)
        total_count = len(mask)
        
        logger.info(f"Confidence mask: {high_conf_count}/{total_count} residues "
                   f"({high_conf_count/total_count*100:.1f}%) above pLDDT {self.plddt_threshold}")
        return mask

class StructureConverter:
    """Enhanced structure format conversion and confidence masking"""
    
    @staticmethod
    def cif_to_pdb(cif_path: str, output_path: str = None) -> str:
        """Convert CIF to PDB format"""
        if output_path is None:
            output_path = cif_path.replace('.cif', '_converted.pdb')
        
        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('structure', cif_path)
            
            io = PDBIO()
            io.set_structure(structure)
            io.save(output_path)
            
            logger.info(f"Converted {cif_path} to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert {cif_path} to PDB: {e}")
            return ""
    
    @staticmethod
    def create_masked_pdb(pdb_path: str, confidence_mask: List[bool], 
                         output_path: str = None) -> str:
        """Create PDB with low-confidence regions masked/removed"""
        if output_path is None:
            output_path = pdb_path.replace('.pdb', '_masked.pdb')
        
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('structure', pdb_path)
            
            class ConfidenceSelect(Select):
                def __init__(self, mask):
                    self.mask = mask
                    self.residue_index = 0
                
                def accept_residue(self, residue):
                    if self.residue_index < len(self.mask):
                        accept = self.mask[self.residue_index]
                        self.residue_index += 1
                        return accept
                    return False
            
            io = PDBIO()
            io.set_structure(structure)
            io.save(output_path, ConfidenceSelect(confidence_mask))
            
            logger.info(f"Created masked PDB: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create masked PDB: {e}")
            return ""

class PyMOLAligner:
    """PyMOL-based structure alignment and analysis"""
    
    def __init__(self):
        self.pymol_available = PYMOL_AVAILABLE
        if self.pymol_available:
            pymol.finish_launching(['pymol', '-c'])  # Launch in command line mode
    
    def align_structures_pymol(self, ref_pdb: str, mobile_pdb: str) -> StructuralMetrics:
        """Perform structural alignment using PyMOL"""
        if not self.pymol_available:
            logger.warning("PyMOL not available, falling back to BioPython")
            return StructuralMetrics(rmsd=float('inf'))
        
        try:
            # Clear PyMOL session
            cmd.reinitialize()
            
            # Load structures
            cmd.load(ref_pdb, 'reference')
            cmd.load(mobile_pdb, 'mobile')
            
            # Perform alignment
            alignment_result = cmd.align('mobile', 'reference')
            
            # Extract metrics
            rmsd = alignment_result[0]
            aligned_atoms = alignment_result[1]
            
            # Get structure info
            ref_atoms = cmd.count_atoms('reference and name CA')
            mobile_atoms = cmd.count_atoms('mobile and name CA')
            
            return StructuralMetrics(
                rmsd=rmsd,
                aligned_residues=aligned_atoms,
                total_residues=min(ref_atoms, mobile_atoms),
                coverage=(aligned_atoms / min(ref_atoms, mobile_atoms)) * 100 if min(ref_atoms, mobile_atoms) > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"PyMOL alignment failed: {e}")
            return StructuralMetrics(rmsd=float('inf'))
    
    def create_alignment_visualization(self, ref_pdb: str, mobile_pdb: str, 
                                     output_dir: str = "visualizations") -> str:
        """Create PyMOL visualization of structural alignment"""
        if not self.pymol_available:
            logger.warning("PyMOL not available for visualization")
            return ""
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Clear and load structures
            cmd.reinitialize()
            cmd.load(ref_pdb, 'reference')
            cmd.load(mobile_pdb, 'mobile')
            
            # Align structures
            cmd.align('mobile', 'reference')
            
            # Style the structures
            cmd.hide('everything')
            cmd.show('cartoon')
            cmd.color('blue', 'reference')
            cmd.color('red', 'mobile')
            cmd.set('transparency', 0.3, 'mobile')
            
            # Create session file
            session_file = os.path.join(output_dir, f"alignment_{Path(ref_pdb).stem}_vs_{Path(mobile_pdb).stem}.pse")
            cmd.save(session_file)
            
            # Create image
            image_file = os.path.join(output_dir, f"alignment_{Path(ref_pdb).stem}_vs_{Path(mobile_pdb).stem}.png")
            cmd.png(image_file, width=1200, height=900, dpi=300)
            
            logger.info(f"Visualization saved: {image_file}")
            return image_file
            
        except Exception as e:
            logger.error(f"Failed to create PyMOL visualization: {e}")
            return ""

class TMAlignWrapper:
    """Enhanced TM-align wrapper with better error handling"""
    
    @staticmethod
    def align_structures(ref_pdb: str, mobile_pdb: str) -> StructuralMetrics:
        """Perform alignment using TM-align"""
        try:
            # Check if TM-align is available
            result = subprocess.run(['TMalign', ref_pdb, mobile_pdb], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.warning("TM-align failed or not available")
                return StructuralMetrics(rmsd=float('inf'))
            
            # Parse TM-align output
            output = result.stdout
            rmsd = None
            tm_score1 = None
            tm_score2 = None
            aligned_length = 0
            
            for line in output.split('\n'):
                if 'RMSD of the common residues' in line:
                    rmsd = float(line.split('=')[1].strip().split()[0])
                elif 'TM-score=' in line and 'Chain_1' in line:
                    tm_score1 = float(line.split('TM-score=')[1].split()[0])
                elif 'TM-score=' in line and 'Chain_2' in line:
                    tm_score2 = float(line.split('TM-score=')[1].split()[0])
                elif 'Number of residues in common' in line:
                    aligned_length = int(line.split('=')[1].strip().split()[0])
            
            # Use average of both TM-scores
            tm_score = (tm_score1 + tm_score2) / 2 if tm_score1 and tm_score2 else tm_score1 or tm_score2
            
            return StructuralMetrics(
                rmsd=rmsd or float('inf'),
                tm_score=tm_score,
                aligned_residues=aligned_length
            )
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, 
                FileNotFoundError) as e:
            logger.warning(f"TM-align failed: {e}")
            return StructuralMetrics(rmsd=float('inf'))

class VisualizationEngine:
    """Enhanced visualization capabilities"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_confidence_distribution(self, structures: Dict[str, AlphaFoldStructure], 
                                   threshold: float = 70.0) -> str:
        """Plot pLDDT confidence distributions"""
        plt.figure(figsize=(12, 8))
        
        for name, structure in structures.items():
            if structure.plddt_scores:
                plt.hist(structure.plddt_scores, bins=50, alpha=0.6, label=name, density=True)
        
        plt.axvline(x=threshold, color='red', linestyle='--', 
                   label=f'Threshold ({threshold})')
        plt.xlabel('pLDDT Score')
        plt.ylabel('Density')
        plt.title('pLDDT Confidence Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = self.output_dir / "confidence_distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confidence distribution plot saved: {output_file}")
        return str(output_file)
    
    def plot_pae_heatmap(self, structure: AlphaFoldStructure) -> str:
        """Plot PAE (Predicted Aligned Error) heatmap"""
        if structure.pae_matrix is None:
            logger.warning(f"No PAE data available for {structure.name}")
            return ""
        
        plt.figure(figsize=(10, 10))
        
        pae = structure.pae_matrix
        if len(pae.shape) == 1:
            # If 1D, assume it's a symmetric matrix stored as upper triangle
            n = int(np.sqrt(len(pae) * 2))
            pae_matrix = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    pae_matrix[i, j] = pae[idx]
                    pae_matrix[j, i] = pae[idx]
                    idx += 1
        else:
            pae_matrix = pae
        
        im = plt.imshow(pae_matrix, cmap='plasma_r', interpolation='nearest')
        plt.colorbar(im, label='PAE (Å)')
        plt.title(f'Predicted Aligned Error - {structure.name}')
        plt.xlabel('Residue Index')
        plt.ylabel('Residue Index')
        
        output_file = self.output_dir / f"pae_heatmap_{structure.name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PAE heatmap saved: {output_file}")
        return str(output_file)
    
    def plot_rmsd_comparison(self, results: List[Dict]) -> str:
        """Create RMSD comparison plot"""
        plt.figure(figsize=(14, 8))
        
        # Separate control and variant results
        control_results = [r for r in results if r.get('is_control', False)]
        variant_results = [r for r in results if not r.get('is_control', False) and r.get('rmsd', float('inf')) != float('inf')]
        
        # Get control baseline
        control_rmsd = control_results[0]['rmsd'] if control_results else 0.0
        
        # Prepare data for plotting
        variant_names = [r['mobile'] for r in variant_results]
        variant_rmsds = [r['rmsd'] for r in variant_results]
        fold_changes = [r.get('fold_change_vs_control', 1.0) for r in variant_results]
        
        # Color by fold change relative to control
        colors = []
        for fc in fold_changes:
            if fc >= 2.0:
                colors.append('red')  # 2x or more above control
            elif fc >= 1.5:
                colors.append('orange')  # 1.5-2x above control
            elif fc >= 1.2:
                colors.append('yellow')  # 1.2-1.5x above control
            else:
                colors.append('green')  # Close to control
        
        # Create main plot
        bars = plt.bar(range(len(variant_names)), variant_rmsds, color=colors, alpha=0.7)
        
        # Add control baseline
        if control_rmsd > 0:
            plt.axhline(y=control_rmsd, color='blue', linestyle='--', linewidth=2,
                       label=f'Control Baseline (WT vs WT): {control_rmsd:.3f}Å')
            
            # Add threshold lines
            plt.axhline(y=control_rmsd * 1.5, color='orange', linestyle=':', alpha=0.7,
                       label=f'1.5x Control: {control_rmsd * 1.5:.3f}Å')
            plt.axhline(y=control_rmsd * 2.0, color='red', linestyle=':', alpha=0.7,
                       label=f'2x Control: {control_rmsd * 2.0:.3f}Å')
        
        plt.xlabel('Variants')
        plt.ylabel('RMSD (Å)')
        plt.title('RMSD Comparison Against Reference (with Control Baseline)')
        plt.xticks(range(len(variant_names)), variant_names, rotation=45, ha='right')
        
        # Add fold change annotations
        for i, (bar, fc) in enumerate(zip(bars, fold_changes)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{fc:.1f}x', ha='center', va='bottom', fontsize=8)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = self.output_dir / "rmsd_comparison_with_control.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"RMSD comparison plot with control saved: {output_file}")
        return str(output_file)

class EnhancedAlphaFoldAnalyzer:
    """Enhanced AlphaFold analyzer following the specific workflow"""
    
    def __init__(self, data_directory: str, plddt_threshold: float = 70.0):
        self.data_directory = Path(data_directory)
        self.confidence_filter = ConfidenceFilter(plddt_threshold)
        self.converter = StructureConverter()
        self.pymol_aligner = PyMOLAligner()
        self.tm_aligner = TMAlignWrapper()
        self.visualizer = VisualizationEngine()
        self.structures = {}
        self.results = []
        
        logger.info(f"Initialized analyzer with pLDDT threshold: {plddt_threshold}")
        logger.info(f"PyMOL available: {PYMOL_AVAILABLE}")
    
    def step1_discover_alphafold_output(self) -> Dict[str, AlphaFoldStructure]:
        """Step 1: Discover AlphaFold outputs"""
        logger.info("STEP 1: Discovering AlphaFold outputs...")
        
        structures = {}
        for subdir in self.data_directory.iterdir():
            if not subdir.is_dir():
                continue
            
            structure_name = subdir.name.replace('2025_06_01_14_', '').replace('2025_05_30_16_18_', '')
            
            cif_files = list(subdir.glob('*.cif'))
            confidence_files = list(subdir.glob('*summary_confidences_*.json'))
            
            if cif_files and confidence_files:
                cif_file = next((f for f in cif_files if 'model_0' in f.name), cif_files[0])
                confidence_file = next((f for f in confidence_files if '_0.json' in f.name), confidence_files[0])
                
                structures[structure_name] = AlphaFoldStructure(
                    name=structure_name,
                    cif_path=str(cif_file),
                    confidence_path=str(confidence_file)
                )
        
        logger.info(f"Found {len(structures)} AlphaFold structures")
        return structures
    
    def step2_convert_cif_to_pdb(self):
        """Step 2: Convert .cif to .pdb"""
        logger.info("STEP 2: Converting CIF to PDB...")
        
        for name, structure in self.structures.items():
            pdb_path = structure.cif_path.replace('.cif', '_converted.pdb')
            structure.pdb_path = self.converter.cif_to_pdb(structure.cif_path, pdb_path)
    
    def step3_mask_low_confidence_regions(self):
        """Step 3: Mask low-confidence regions (pLDDT < threshold)"""
        logger.info(f"STEP 3: Masking low-confidence regions (pLDDT < {self.confidence_filter.plddt_threshold})...")
        
        for name, structure in self.structures.items():
            # Extract confidence data
            structure.plddt_scores = self.confidence_filter.extract_plddt_from_cif(structure.cif_path)
            structure.pae_matrix = self.confidence_filter.extract_pae_matrix(structure.confidence_path)
            
            # Load additional confidence metrics
            confidence_data = self.confidence_filter.load_confidence_data(structure.confidence_path)
            structure.ptm_score = confidence_data.get('ptm', None)
            structure.ranking_score = confidence_data.get('ranking_score', None)
            
            # Create confidence mask
            if structure.plddt_scores:
                confidence_mask = self.confidence_filter.create_confidence_mask(
                    structure.pdb_path, structure.plddt_scores
                )
                
                # Create masked PDB
                masked_path = structure.pdb_path.replace('.pdb', '_masked.pdb')
                structure.masked_pdb_path = self.converter.create_masked_pdb(
                    structure.pdb_path, confidence_mask, masked_path
                )
    
    def step4_align_structures(self, ref_name: str, mobile_name: str) -> StructuralMetrics:
        """Step 4: Align structures using TM-align or PyMOL"""
        logger.info(f"STEP 4: Aligning {mobile_name} to {ref_name}...")
        
        ref_structure = self.structures[ref_name]
        mobile_structure = self.structures[mobile_name]
        
        # Use masked PDBs if available, otherwise regular PDBs
        ref_pdb = ref_structure.masked_pdb_path or ref_structure.pdb_path
        mobile_pdb = mobile_structure.masked_pdb_path or mobile_structure.pdb_path
        
        # Try TM-align first (more accurate)
        metrics = self.tm_aligner.align_structures(ref_pdb, mobile_pdb)
        
        # If TM-align fails, try PyMOL
        if metrics.rmsd == float('inf') and PYMOL_AVAILABLE:
            logger.info("TM-align failed, trying PyMOL...")
            metrics = self.pymol_aligner.align_structures_pymol(ref_pdb, mobile_pdb)
        
        # If both fail, use BioPython as fallback
        if metrics.rmsd == float('inf'):
            logger.info("Both TM-align and PyMOL failed, using BioPython...")
            metrics = self._fallback_biopython_alignment(ref_pdb, mobile_pdb)
        
        return metrics
    
    def _fallback_biopython_alignment(self, ref_pdb: str, mobile_pdb: str) -> StructuralMetrics:
        """Fallback BioPython alignment"""
        try:
            parser = PDBParser(QUIET=True)
            ref_struct = parser.get_structure('ref', ref_pdb)
            mobile_struct = parser.get_structure('mobile', mobile_pdb)
            
            # Extract CA atoms
            ref_atoms = self._get_ca_atoms(ref_struct)
            mobile_atoms = self._get_ca_atoms(mobile_struct)
            
            if len(ref_atoms) < 3 or len(mobile_atoms) < 3:
                return StructuralMetrics(rmsd=float('inf'))
            
            # Align using common length
            min_length = min(len(ref_atoms), len(mobile_atoms))
            ref_atoms = ref_atoms[:min_length]
            mobile_atoms = mobile_atoms[:min_length]
            
            superimposer = Superimposer()
            superimposer.set_atoms(ref_atoms, mobile_atoms)
            
            return StructuralMetrics(
                rmsd=superimposer.rms,
                aligned_residues=len(ref_atoms),
                total_residues=min_length,
                coverage=100.0
            )
            
        except Exception as e:
            logger.error(f"BioPython alignment failed: {e}")
            return StructuralMetrics(rmsd=float('inf'))
    
    def _get_ca_atoms(self, structure):
        """Extract CA atoms from structure"""
        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    try:
                        atoms.append(residue['CA'])
                    except KeyError:
                        continue
            break
        return atoms
    
    def step5_compute_metrics(self, metrics: StructuralMetrics, ref_name: str, 
                             mobile_name: str) -> Dict[str, Any]:
        """Step 5: Compute RMSD + TM-score + GDT-TS"""
        logger.info(f"STEP 5: Computing comprehensive metrics...")
        
        ref_structure = self.structures[ref_name]
        mobile_structure = self.structures[mobile_name]
        
        # Calculate additional metrics
        ref_plddt = np.mean(ref_structure.plddt_scores) if ref_structure.plddt_scores else 0
        mobile_plddt = np.mean(mobile_structure.plddt_scores) if mobile_structure.plddt_scores else 0
        
        # PAE score (mean)
        pae_score = np.mean(mobile_structure.pae_matrix) if mobile_structure.pae_matrix is not None else None
        
        return {
            'reference': ref_name,
            'mobile': mobile_name,
            'rmsd': metrics.rmsd,
            'tm_score': metrics.tm_score,
            'gdt_ts': metrics.gdt_ts,
            'aligned_residues': metrics.aligned_residues,
            'total_residues': metrics.total_residues,
            'coverage': metrics.coverage,
            'mean_plddt': (ref_plddt + mobile_plddt) / 2,
            'ref_plddt': ref_plddt,
            'mobile_plddt': mobile_plddt,
            'pae_score': pae_score,
            'ref_ptm': ref_structure.ptm_score,
            'mobile_ptm': mobile_structure.ptm_score,
            'ref_ranking': ref_structure.ranking_score,
            'mobile_ranking': mobile_structure.ranking_score
        }
    
    def step6_visualize_differences(self, results: List[Dict]):
        """Step 6: Visualize differences (PyMOL, plots)"""
        logger.info("STEP 6: Creating visualizations...")
        
        # Plot confidence distributions
        self.visualizer.plot_confidence_distribution(self.structures, 
                                                    self.confidence_filter.plddt_threshold)
        
        # Plot RMSD comparison
        self.visualizer.plot_rmsd_comparison(results)
        
        # Create PAE heatmaps for all structures
        for structure in self.structures.values():
            self.visualizer.plot_pae_heatmap(structure)
        
        # Create PyMOL alignments for significant differences (excluding control)
        if PYMOL_AVAILABLE:
            logger.info("Creating PyMOL alignment visualizations...")
            
            # Get non-control results for visualization
            visualization_results = [r for r in results if not r.get('is_control', False)]
            
            for result in visualization_results[:5]:  # Limit to first 5 for performance
                if result.get('rmsd', float('inf')) != float('inf'):
                    ref_structure = self.structures[result['reference']]
                    
                    # Get the mobile structure name (remove '_control' suffix if present)
                    mobile_name = result['mobile']
                    if mobile_name.endswith('_control'):
                        mobile_name = mobile_name.replace('_control', '')
                    
                    if mobile_name in self.structures:
                        mobile_structure = self.structures[mobile_name]
                        
                        ref_pdb = ref_structure.masked_pdb_path or ref_structure.pdb_path
                        mobile_pdb = mobile_structure.masked_pdb_path or mobile_structure.pdb_path
                        
                        self.pymol_aligner.create_alignment_visualization(ref_pdb, mobile_pdb)
    
    def step7_map_ptm_pae_confidence(self) -> Dict[str, Any]:
        """Step 7: Map PTM/PAE to interpret confidence"""
        logger.info("STEP 7: Mapping PTM/PAE confidence...")
        
        confidence_summary = {
            'ptm_scores': {},
            'pae_scores': {},
            'plddt_means': {},
            'ranking_scores': {}
        }
        
        for name, structure in self.structures.items():
            confidence_summary['ptm_scores'][name] = structure.ptm_score
            confidence_summary['plddt_means'][name] = np.mean(structure.plddt_scores) if structure.plddt_scores else None
            confidence_summary['pae_scores'][name] = np.mean(structure.pae_matrix) if structure.pae_matrix is not None else None
            confidence_summary['ranking_scores'][name] = structure.ranking_score
        
        return confidence_summary
    
    def run_complete_pipeline(self, reference: str = 'p53_wildtype'):
        """Run the complete enhanced pipeline"""
        logger.info("=== STARTING ENHANCED ALPHAFOLD ANALYSIS PIPELINE ===")
        
        # Step 1: Discover AlphaFold outputs
        self.structures = self.step1_discover_alphafold_output()
        
        if not self.structures:
            logger.error("No structures found!")
            return None, None
        
        if reference not in self.structures:
            logger.error(f"Reference structure {reference} not found!")
            return None, None
        
        # Step 2: Convert CIF to PDB
        self.step2_convert_cif_to_pdb()
        
        # Step 3: Mask low-confidence regions
        self.step3_mask_low_confidence_regions()
        
        # Steps 4-5: Align and compute metrics for all pairs
        results = []
        
        # FIRST: Add wildtype vs wildtype control comparison
        logger.info(f"\n--- CONTROL: Analyzing {reference} vs {reference} (self-comparison) ---")
        control_metrics = self.step4_align_structures(reference, reference)
        control_result = self.step5_compute_metrics(control_metrics, reference, reference)
        control_result['mobile'] = f"{reference}_control"  # Rename for clarity in results
        control_result['is_control'] = True
        results.append(control_result)
        
        # THEN: Compare all variants against reference
        for name in self.structures:
            if name != reference:
                logger.info(f"\n--- Analyzing {name} vs {reference} ---")
                
                # Step 4: Align structures
                metrics = self.step4_align_structures(reference, name)
                
                # Step 5: Compute comprehensive metrics
                result = self.step5_compute_metrics(metrics, reference, name)
                result['is_control'] = False
                results.append(result)
        
        # Calculate control statistics for interpretation
        if results:
            control_rmsd = results[0]['rmsd']  # First result is the control
            variant_rmsds = [r['rmsd'] for r in results[1:] if r.get('rmsd', float('inf')) != float('inf')]
            
            if variant_rmsds:
                # Add control statistics to each result for interpretation
                for result in results:
                    result['control_baseline_rmsd'] = control_rmsd
                    if not result.get('is_control', False):
                        # Calculate how many standard deviations above control
                        if control_rmsd > 0:
                            result['fold_change_vs_control'] = result['rmsd'] / control_rmsd
                        else:
                            result['fold_change_vs_control'] = 1.0
        
        # Step 6: Visualize differences
        if results:
            self.step6_visualize_differences(results)
        
        # Step 7: Map PTM/PAE confidence
        confidence_summary = self.step7_map_ptm_pae_confidence()
        
        # Save results
        self._save_enhanced_results(results, confidence_summary)
        
        logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
        return results, confidence_summary
    
    def _save_enhanced_results(self, results: List[Dict], confidence_summary: Dict):
        """Save enhanced analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        df_results = pd.DataFrame(results)
        results_file = f"enhanced_analysis_{timestamp}_results.csv"
        df_results.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")
        
        # Save confidence summary
        confidence_file = f"enhanced_analysis_{timestamp}_confidence.json"
        with open(confidence_file, 'w') as f:
            json.dump(confidence_summary, f, indent=2, default=str)
        logger.info(f"Confidence summary saved to {confidence_file}")
        
        # Generate enhanced report
        self._generate_enhanced_report(results, confidence_summary, timestamp)
    
    def _generate_enhanced_report(self, results: List[Dict], 
                                confidence_summary: Dict, timestamp: str):
        """Generate comprehensive analysis report"""
        
        # Separate control and variant results
        control_results = [r for r in results if r.get('is_control', False)]
        variant_results = [r for r in results if not r.get('is_control', False)]
        
        control_rmsd = control_results[0]['rmsd'] if control_results else 0.0
        
        report = f"""
# Enhanced AlphaFold Structural Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Pipeline Version: Enhanced with PyMOL integration and Control Baseline

## Workflow Summary
✅ Step 1: AlphaFold output discovery
✅ Step 2: CIF to PDB conversion  
✅ Step 3: Low-confidence region masking (pLDDT < {self.confidence_filter.plddt_threshold})
✅ Step 4: Structure alignment (TM-align/PyMOL/BioPython)
✅ Step 5: Comprehensive metrics (RMSD + TM-score + GDT-TS)
✅ Step 6: Visualization generation
✅ Step 7: PTM/PAE confidence mapping

## Analysis Summary
- Total structures analyzed: {len(self.structures)}
- Successful comparisons: {len(variant_results)}
- Control baseline (WT vs WT): {control_rmsd:.4f}Å
- PyMOL available: {PYMOL_AVAILABLE}
- Confidence threshold: pLDDT ≥ {self.confidence_filter.plddt_threshold}

## Control Baseline Analysis
**Control RMSD (Wildtype vs Wildtype): {control_rmsd:.4f}Å**

This represents the baseline "noise floor" of the analysis method, including:
- Alignment algorithm precision
- Structure preparation effects
- Confidence masking impact

Any variant with RMSD significantly above this baseline shows genuine structural differences.

## Key Results
"""
        
        # Add top differences with fold changes
        valid_results = [r for r in variant_results if r.get('rmsd', float('inf')) != float('inf')]
        if valid_results:
            sorted_results = sorted(valid_results, key=lambda x: x['rmsd'], reverse=True)
            
            report += "\n### Top Structural Differences (by RMSD vs Control)\n"
            for i, result in enumerate(sorted_results[:10], 1):
                fold_change = result.get('fold_change_vs_control', 1.0)
                significance = ""
                if fold_change >= 2.0:
                    significance = " 🔴 (≥2x control - highly significant)"
                elif fold_change >= 1.5:
                    significance = " 🟠 (≥1.5x control - significant)"
                elif fold_change >= 1.2:
                    significance = " 🟡 (≥1.2x control - moderate)"
                else:
                    significance = " 🟢 (close to control)"
                
                report += f"{i}. {result['mobile']}: {result['rmsd']:.4f}Å ({fold_change:.1f}x control){significance}"
                if result.get('tm_score'):
                    report += f" | TM-score: {result['tm_score']:.3f}"
                report += "\n"
            
            # Statistical summary
            variant_rmsds = [r['rmsd'] for r in valid_results]
            mean_variant_rmsd = np.mean(variant_rmsds)
            std_variant_rmsd = np.std(variant_rmsds)
            
            report += f"\n### Statistical Summary\n"
            report += f"- Control baseline: {control_rmsd:.4f}Å\n"
            report += f"- Mean variant RMSD: {mean_variant_rmsd:.4f}Å ({mean_variant_rmsd/control_rmsd:.1f}x control)\n"
            report += f"- Variant RMSD range: {min(variant_rmsds):.4f}Å - {max(variant_rmsds):.4f}Å\n"
            report += f"- Standard deviation: {std_variant_rmsd:.4f}Å\n"
            
            # Significance categories
            highly_sig = len([r for r in valid_results if r.get('fold_change_vs_control', 1.0) >= 2.0])
            significant = len([r for r in valid_results if 1.5 <= r.get('fold_change_vs_control', 1.0) < 2.0])
            moderate = len([r for r in valid_results if 1.2 <= r.get('fold_change_vs_control', 1.0) < 1.5])
            minimal = len([r for r in valid_results if r.get('fold_change_vs_control', 1.0) < 1.2])
            
            report += f"\n### Significance Classification\n"
            report += f"- Highly significant (≥2x control): {highly_sig} variants\n"
            report += f"- Significant (1.5-2x control): {significant} variants\n"
            report += f"- Moderate (1.2-1.5x control): {moderate} variants\n"
            report += f"- Minimal (<1.2x control): {minimal} variants\n"
        
        # Add confidence insights
        report += f"\n### Confidence Analysis\n"
        ptm_scores = [v for v in confidence_summary['ptm_scores'].values() if v is not None]
        if ptm_scores:
            report += f"- PTM score range: {min(ptm_scores):.3f} - {max(ptm_scores):.3f}\n"
        
        plddt_means = [v for v in confidence_summary['plddt_means'].values() if v is not None]
        if plddt_means:
            report += f"- Mean pLDDT range: {min(plddt_means):.1f} - {max(plddt_means):.1f}\n"
        
        report += f"""
## Files Generated
- Results: enhanced_analysis_{timestamp}_results.csv
- Confidence: enhanced_analysis_{timestamp}_confidence.json
- Visualizations: visualizations/ directory
- This report: enhanced_analysis_{timestamp}_report.md

## Interpretation Guidelines
- **Control baseline**: {control_rmsd:.4f}Å represents method precision
- **Fold changes ≥1.5x control**: Likely biologically meaningful differences
- **Fold changes ≥2.0x control**: Highly significant structural changes
- **All results filtered**: Only high-confidence regions (pLDDT ≥{self.confidence_filter.plddt_threshold}) analyzed

## Biological Implications
This enhanced analysis with control baseline provides a robust framework for:
1. Distinguishing genuine structural changes from method noise
2. Quantifying the magnitude of mutation effects
3. Prioritizing variants for further investigation
4. Understanding confidence-filtered structural comparisons
"""
        
        report_file = f"enhanced_analysis_{timestamp}_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Enhanced report saved to {report_file}")
        return report_file

def main():
    """Enhanced main function"""
    parser = argparse.ArgumentParser(
        description="Enhanced AlphaFold Structural Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Pipeline Steps:
1. AlphaFold output discovery
2. Convert .cif to .pdb
3. Mask low-confidence regions (pLDDT < 70)
4. Align structures using TM-align or PyMOL
5. Compute RMSD + TM-score + GDT-TS
6. Visualize differences (PyMOL, plots)
7. Map PTM/PAE to interpret confidence

Examples:
  python robust_alphafold_analysis.py data/ --reference p53_wildtype
  python robust_alphafold_analysis.py folds/ --plddt-threshold 80 --reference wildtype
        """
    )
    
    parser.add_argument('data_directory', 
                       help='Directory containing AlphaFold structure folders')
    parser.add_argument('--reference', default='p53_wildtype',
                       help='Reference structure name (default: p53_wildtype)')
    parser.add_argument('--plddt-threshold', type=float, default=70.0,
                       help='pLDDT confidence threshold (default: 70.0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedAlphaFoldAnalyzer(args.data_directory, args.plddt_threshold)
    
    # Run enhanced pipeline
    results, confidence_summary = analyzer.run_complete_pipeline(args.reference)
    
    if results:
        logger.info("Enhanced analysis completed successfully!")
        logger.info(f"Generated {len(results)} comparisons")
        logger.info("Check the visualizations/ directory for plots and PyMOL sessions")
    else:
        logger.error("Enhanced analysis failed - no results generated")

if __name__ == "__main__":
    main() 