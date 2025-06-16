# AlphaFold Structural Analysis Script: Complete Workflow Explanation

## Overview

The `robust_alphafold_analysis.py` script is a comprehensive pipeline designed to compare AlphaFold-predicted protein structures, specifically for analyzing the structural impact of missense mutations on p53 protein variants. This document explains each step of the workflow and the rationale behind the design decisions.

---

## Architecture Overview

### Design Philosophy

The script follows a **modular, object-oriented design** with the following principles:

1. **Separation of Concerns**: Each class handles a specific aspect (confidence filtering, structure conversion, alignment, visualization)
2. **Robustness**: Multiple fallback methods for critical operations (TM-align → PyMOL → BioPython)
3. **Confidence-First**: All analysis filtered by AlphaFold confidence scores (pLDDT) to ensure reliability
4. **Control Baseline**: Includes wildtype-vs-wildtype comparison to establish noise floor
5. **Comprehensive Output**: Multiple formats (CSV, JSON, markdown reports, visualizations)

### Key Classes and Their Roles

```python
@dataclass
class StructuralMetrics:
    """Container for structural comparison metrics"""
    rmsd: float
    tm_score: Optional[float] = None
    gdt_ts: Optional[float] = None
    # ... other metrics
```

**Design Decision**: Using dataclasses for clean, type-safe data containers that are easy to extend and debug.

```python
@dataclass
class AlphaFoldStructure:
    """Container for AlphaFold structure data"""
    name: str
    cif_path: str
    pdb_path: Optional[str] = None
    masked_pdb_path: Optional[str] = None
    # ... confidence data
```

**Design Decision**: Centralizing all structure-related metadata in one container to avoid parameter passing complexity and ensure data consistency.

---

## Step-by-Step Workflow Explanation

### Step 1: AlphaFold Output Discovery

**Location**: `step1_discover_alphafold_output()`

```python
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
```

**What it does**:

- Scans the input directory for AlphaFold output folders
- Each folder should contain `.cif` structure files and `*summary_confidences*.json` files
- Creates `AlphaFoldStructure` objects to track file paths and metadata

**Design Decisions**:

1. **Automatic Discovery**: Instead of requiring manual file lists, the script automatically finds relevant files using glob patterns
2. **Flexible Naming**: The script handles different timestamp prefixes in folder names by stripping them off
3. **Paired Files**: Ensures both structure (.cif) AND confidence (.json) files are present before proceeding
4. **Model Selection**: Prefers "model_0" files when multiple models exist (highest confidence)

**Why this approach**:

- **Robustness**: Handles variations in AlphaFold output naming conventions
- **Automation**: Reduces manual setup and potential errors
- **Validation**: Early detection of missing or malformed data

### Step 2: CIF to PDB Conversion

**Location**: `step2_convert_cif_to_pdb()` → `StructureConverter.cif_to_pdb()`

```python
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
```

**What it does**:

- Converts AlphaFold's native CIF format to PDB format
- Uses BioPython's MMCIFParser and PDBIO for reliable conversion

**Design Decisions**:

1. **Format Standardization**: Many downstream tools (TM-align, PyMOL) work better with PDB format
2. **BioPython Usage**: Leverages well-tested, widely-used library for structural biology
3. **Error Handling**: Graceful failure with logging if conversion fails
4. **Automatic Naming**: Generates output names systematically to avoid conflicts

**Why this approach**:

- **Compatibility**: PDB format has broader tool support than CIF
- **Reliability**: BioPython handles format nuances and edge cases
- **Traceability**: Maintains clear naming conventions for intermediate files

### Step 3: Confidence-Based Masking

**Location**: `step3_mask_low_confidence_regions()` → `ConfidenceFilter`

```python
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
```

**What it does**:

- Extracts per-residue confidence scores (pLDDT) from the B-factor field in CIF files
- Creates a boolean mask for residues above the confidence threshold (default: 70.0)
- Generates "masked" PDB files containing only high-confidence regions

**Design Decisions**:

1. **Conservative Threshold**: Default pLDDT ≥ 70.0 is considered "confident" by AlphaFold standards
2. **Per-Residue Filtering**: More granular than per-domain filtering
3. **Physical Removal**: Creates new PDB files rather than just flagging, ensuring downstream tools only see reliable regions
4. **CA-Based Extraction**: Uses Cα atoms as representative of residue confidence

**Why this approach**:

- **Scientific Rigor**: Eliminates potentially unreliable structural regions from analysis
- **AlphaFold Best Practice**: Follows recommended confidence thresholds from DeepMind
- **Downstream Safety**: Prevents alignment algorithms from being misled by low-confidence regions
- **Transparency**: Logs exactly how much data is retained vs. filtered

### Step 4: Structure Alignment with Fallback Strategy

**Location**: `step4_align_structures()` with multiple aligner classes

```python
def step4_align_structures(self, ref_name: str, mobile_name: str) -> StructuralMetrics:
    """Step 4: Align structures using TM-align or PyMOL"""
    # Try TM-align first (most accurate)
    metrics = self.tm_aligner.align_structures(ref_pdb, mobile_pdb)

    # If TM-align fails, try PyMOL
    if metrics.rmsd == float('inf') and PYMOL_AVAILABLE:
        logger.info("TM-align failed, trying PyMOL...")
        metrics = self.pymol_aligner.align_structures_pymol(ref_pdb, mobile_pdb)

    # If both fail, use BioPython as fallback
    if metrics.rmsd == float('inf'):
        logger.info("Both TM-align and PyMOL failed, using BioPython...")
        metrics = self._fallback_biopython_alignment(ref_pdb, mobile_pdb)
```

**What it does**:

- Attempts structural alignment using three different methods in order of preference
- Returns comprehensive metrics including RMSD, TM-score, and alignment coverage

**Design Decisions**:

1. **Hierarchical Fallback**: TM-align (best) → PyMOL (good) → BioPython (basic but reliable)
2. **Tool Detection**: Automatically detects if PyMOL is available and adapts accordingly
3. **Consistent Interface**: All aligners return the same `StructuralMetrics` dataclass
4. **Comprehensive Metrics**: Not just RMSD, but also TM-score and GDT-TS when available

**Why this approach**:

- **Maximizes Success Rate**: If one method fails, others provide backup
- **Quality Optimization**: Uses the best available tool while maintaining compatibility
- **Research Grade**: TM-align is gold standard for protein structure comparison
- **Practical**: BioPython fallback ensures the pipeline never completely fails

#### TM-align Integration

```python
@staticmethod
def align_structures(ref_pdb: str, mobile_pdb: str) -> StructuralMetrics:
    """Align structures using TM-align"""
    try:
        # Run TM-align as subprocess
        result = subprocess.run([
            'TMalign', ref_pdb, mobile_pdb
        ], capture_output=True, text=True, timeout=300)

        # Parse output for metrics
        rmsd = TMAlignWrapper._extract_rmsd(result.stdout)
        tm_score = TMAlignWrapper._extract_tm_score(result.stdout)
```

**Design Decision**: Uses subprocess to call external TM-align binary rather than Python bindings.

**Rationale**:

- TM-align is a C++ program with no reliable Python bindings
- Subprocess approach is more robust and handles edge cases better
- Allows timeout to prevent hanging on problematic structures

### Step 5: Comprehensive Metrics Calculation

**Location**: `step5_compute_metrics()`

```python
def step5_compute_metrics(self, metrics: StructuralMetrics, ref_name: str,
                         mobile_name: str) -> Dict[str, Any]:
    """Step 5: Compute RMSD + TM-score + GDT-TS"""
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
        # ... confidence and coverage metrics
    }
```

**What it does**:

- Combines structural alignment metrics with confidence scores
- Calculates summary statistics for interpretation
- Creates a comprehensive result dictionary for each comparison

**Design Decisions**:

1. **Multi-Modal Metrics**: Combines structural similarity (RMSD, TM-score) with confidence (pLDDT, PAE)
2. **Summary Statistics**: Mean pLDDT and PAE provide overall quality assessment
3. **Structured Output**: Consistent dictionary format for easy downstream processing
4. **Confidence Tracking**: Maintains both reference and mobile structure confidence scores

**Why this approach**:

- **Holistic Assessment**: Structural similarity alone isn't sufficient; confidence matters
- **Quality Control**: Low-confidence alignments can be flagged and handled appropriately
- **Research Standards**: Follows best practices in structural biology for comparative analysis

### Step 6: Visualization Generation

**Location**: `step6_visualize_differences()` → `VisualizationEngine`

```python
def plot_rmsd_comparison(self, results: List[Dict]) -> str:
    """Create RMSD comparison plot"""
    # Separate control vs variants
    control_results = [r for r in results if r.get('is_control', False)]
    variant_results = [r for r in results if not r.get('is_control', False)]

    # Create publication-quality plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: RMSD bar chart
    # Plot 2: Fold change vs control
```

**What it does**:

- Creates multiple types of visualizations (bar charts, heatmaps, confidence distributions)
- Generates PyMOL session files for interactive 3D visualization
- Produces publication-quality figures with proper labeling and statistics

**Design Decisions**:

1. **Multiple Formats**: Both static plots (PNG) and interactive sessions (PyMOL)
2. **Publication Ready**: Professional formatting, clear legends, statistical annotations
3. **Modular Generation**: Separate methods for different plot types
4. **Control Highlighting**: Special treatment for control comparisons in visualizations

**Why this approach**:

- **Scientific Communication**: Visualizations are crucial for understanding and presenting results
- **Quality Assurance**: Visual inspection can reveal issues not apparent in numerical data
- **Interactive Analysis**: PyMOL sessions allow detailed exploration of structural differences

### Step 7: Confidence Mapping and Integration

**Location**: `step7_map_ptm_pae_confidence()`

```python
def step7_map_ptm_pae_confidence(self) -> Dict[str, Any]:
    """Step 7: Map PTM/PAE to interpret confidence"""
    confidence_summary = {
        'plddt_means': {},
        'ptm_scores': {},
        'pae_means': {},
        'confidence_stats': {}
    }

    for name, structure in self.structures.items():
        if structure.plddt_scores:
            confidence_summary['plddt_means'][name] = np.mean(structure.plddt_scores)
```

**What it does**:

- Aggregates confidence metrics across all structures
- Creates summary statistics for confidence assessment
- Maps confidence scores to structural analysis results

**Design Decisions**:

1. **Comprehensive Coverage**: Includes all available confidence metrics (pLDDT, PTM, PAE)
2. **Statistical Summary**: Both per-structure and aggregate statistics
3. **JSON Output**: Machine-readable format for further analysis
4. **Integration Ready**: Designed to be combined with structural results

**Why this approach**:

- **Confidence-Aware Analysis**: Essential for interpreting AlphaFold predictions correctly
- **Quality Documentation**: Provides clear record of prediction confidence
- **Future Analysis**: Enables post-hoc confidence-based filtering or weighting

### Complete Pipeline Execution

**Location**: `run_complete_pipeline()`

```python
def run_complete_pipeline(self, reference: str = 'p53_wildtype'):
    """Run the complete enhanced pipeline"""
    logger.info("=== STARTING ENHANCED ALPHAFOLD ANALYSIS PIPELINE ===")

    # Step 1: Discover AlphaFold outputs
    self.structures = self.step1_discover_alphafold_output()

    # Steps 2-3: Convert and mask
    self.step2_convert_cif_to_pdb()
    self.step3_mask_low_confidence_regions()

    # Steps 4-5: Align and compute metrics for all pairs
    results = []

    # FIRST: Add wildtype vs wildtype control comparison
    logger.info(f"\n--- CONTROL: Analyzing {reference} vs {reference} (self-comparison) ---")
    control_metrics = self.step4_align_structures(reference, reference)
    control_result = self.step5_compute_metrics(control_metrics, reference, reference)
    control_result['is_control'] = True
    results.append(control_result)

    # THEN: Compare all variants against reference
    for name in self.structures:
        if name != reference:
            # Steps 4-5 for each variant...
```

**What it does**:

- Orchestrates the complete 7-step workflow
- Implements control baseline establishment
- Manages error handling and result collection
- Generates comprehensive output files

**Key Design Decisions**:

1. **Control-First Approach**: Always performs wildtype-vs-wildtype comparison first to establish baseline noise
2. **Reference-Centric**: All comparisons are made against a single reference structure
3. **Error Resilience**: Individual comparison failures don't stop the entire pipeline
4. **Comprehensive Output**: Multiple output formats for different use cases

**Why this approach**:

- **Scientific Rigor**: Control baseline is essential for interpreting structural differences
- **Batch Processing**: Efficiently handles multiple variants in a single run
- **Fault Tolerance**: Maximizes useful output even when some comparisons fail
- **Flexibility**: Modular design allows easy modification for different analysis needs

---

## Critical Design Decisions and Rationale

### 1. Confidence-First Philosophy

**Decision**: Filter all analysis by pLDDT ≥ 70.0 confidence threshold.

**Rationale**:

- AlphaFold predictions vary dramatically in reliability
- Low-confidence regions can dominate RMSD calculations and mislead interpretation
- pLDDT 70+ is considered "confident" by AlphaFold standards
- Better to analyze fewer, reliable residues than include unreliable ones

### 2. Multiple Alignment Methods with Fallback

**Decision**: TM-align → PyMOL → BioPython hierarchical approach.

**Rationale**:

- TM-align is research gold standard but may fail on unusual structures
- PyMOL provides robust alignment with good visualization integration
- BioPython ensures the pipeline never completely fails
- Different methods may be optimal for different structure types

### 3. Control Baseline Establishment

**Decision**: Always include wildtype-vs-wildtype comparison as control.

**Rationale**:

- Establishes "noise floor" of the analysis method
- Enables statistical interpretation of variant differences
- Controls for alignment precision, file conversion artifacts, and masking effects
- Provides reference point for determining biological significance

### 4. Comprehensive Output Strategy

**Decision**: Generate CSV, JSON, markdown reports, and visualizations.

**Rationale**:

- CSV for quantitative analysis and statistics
- JSON for machine-readable confidence data
- Markdown for human-readable interpretation
- Visualizations for presentation and quality control
- Multiple formats serve different downstream needs

### 5. Object-Oriented Modular Design

**Decision**: Separate classes for different functionality areas.

**Rationale**:

- Easier testing and debugging
- Enables reuse of components in other projects
- Clear separation of concerns
- Facilitates collaborative development
- Makes the codebase more maintainable

### 6. Robust Error Handling

**Decision**: Graceful degradation rather than hard failures.

**Rationale**:

- Structural biology data can be messy and unpredictable
- Better to get partial results than no results
- Enables batch processing of large datasets
- Provides actionable error information for troubleshooting

---

## Usage and Command Line Interface

```bash
# Basic usage
python robust_alphafold_analysis.py folds_2025_06_01_09_25/ --reference p53_wildtype

# With custom confidence threshold
python robust_alphafold_analysis.py data/ --plddt-threshold 80 --reference wildtype

# Verbose output for debugging
python robust_alphafold_analysis.py data/ --verbose --reference p53_wildtype
```

The script is designed to be both powerful for expert users and accessible for routine analysis, with sensible defaults and clear documentation for all parameters.

---

## Output Files and Interpretation

1. **`enhanced_analysis_TIMESTAMP_results.csv`**: Main quantitative results
2. **`enhanced_analysis_TIMESTAMP_confidence.json`**: Confidence metrics
3. **`enhanced_analysis_TIMESTAMP_report.md`**: Human-readable interpretation
4. **`visualizations/`**: Directory containing plots and PyMOL sessions

This comprehensive design ensures the script serves as both a research tool and a robust production pipeline for AlphaFold structure comparison analysis.
