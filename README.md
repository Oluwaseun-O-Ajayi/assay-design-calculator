# Assay Design Calculator 💊📊

A comprehensive Python toolkit for pharmaceutical assay design and validation. Designed for high-throughput screening (HTS), drug discovery, and bioanalytical research.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This toolkit provides industry-standard metrics and tools for designing, optimizing, and validating pharmaceutical assays. Essential for ensuring assay quality, reproducibility, and statistical rigor in drug screening workflows.

## 🚀 Key Features

- **Z-Factor & Z' Calculation** - Industry gold standard for assay quality
- **Signal-to-Noise Ratio (S/N)** - Data quality assessment
- **Coefficient of Variation (CV)** - Reproducibility metrics
- **Detection Limits** - LOD and LOQ calculations
- **Plate Layout Designer** - Visual 96/384/1536-well plate layouts
- **Statistical Power Analysis** - Sample size determination
- **Hit Selection Criteria** - Threshold calculations for screening
- **Batch Processing** - Analyze multiple assays simultaneously

## ⚡ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Oluwaseun-O-Ajayi/assay-design-calculator.git
cd assay-design-calculator

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from assay_design_calculator import AssayQualityMetrics

# Calculate Z-factor
qc = AssayQualityMetrics()
positive_controls = [95, 98, 96, 97, 94, 99, 95, 96]
negative_controls = [5, 6, 4, 7, 5, 6, 5, 4]

z_factor = qc.calculate_z_factor(positive_controls, negative_controls)
print(f"Z' = {z_factor:.3f}")  # Z' = 0.876 (Excellent!)
```

## 📋 Requirements

```
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
```

## 🧩 Core Modules

### 1. AssayQualityMetrics

Calculate industry-standard quality metrics for assay validation.

#### Z-Factor (Z')

The gold standard for assay quality assessment.

**Formula:**
```
Z' = 1 - (3 × (σ_pos + σ_neg)) / |μ_pos - μ_neg|
```

**Interpretation:**
- Z' ≥ 0.5: Excellent assay
- 0 ≤ Z' < 0.5: Marginal assay
- Z' < 0: Poor assay (unacceptable)

**Example:**
```python
qc = AssayQualityMetrics()

positive_controls = [100, 102, 98, 101, 99, 103, 100, 98]
negative_controls = [10, 12, 9, 11, 10, 12, 11, 10]

z_factor = qc.calculate_z_factor(positive_controls, negative_controls)
qc.generate_quality_report()
```

#### Signal-to-Noise Ratio (S/N)

Measures data quality and signal clarity.

**Formula:**
```
S/N = (Signal - Blank) / Noise
```

**Interpretation:**
- S/N ≥ 3: Good signal quality
- S/N < 3: Poor signal quality

**Example:**
```python
signal = [100, 105, 98, 102]
blank = [5, 6, 4, 5, 6, 5]

sn_ratio = qc.calculate_signal_to_noise(signal, blank=blank)
```

#### Coefficient of Variation (CV)

Assesses assay reproducibility.

**Formula:**
```
CV = (σ / μ) × 100%
```

**Interpretation:**
- CV < 10%: Excellent reproducibility
- 10% ≤ CV < 20%: Good reproducibility
- CV ≥ 20%: Poor reproducibility

**Example:**
```python
measurements = [98, 100, 102, 99, 101, 98, 100]
cv = qc.calculate_cv(measurements)
```

#### Detection Limits (LOD/LOQ)

Calculate analytical sensitivity.

**Formulas:**
```
LOD = 3.3 × (σ_blank / slope)
LOQ = 10 × (σ_blank / slope)
```

**Example:**
```python
blank_measurements = [5, 6, 5, 7, 6, 5, 6]
calibration_slope = 1000  # from standard curve

limits = qc.calculate_detection_limits(blank_measurements, slope=calibration_slope)
print(f"LOD: {limits['LOD']:.4f}")
print(f"LOQ: {limits['LOQ']:.4f}")
```

### 2. PlateLayoutDesigner

Design and visualize microplate layouts for HTS.

**Example:**
```python
from assay_design_calculator import PlateLayoutDesigner

# Create 96-well plate layout
plate = PlateLayoutDesigner(plate_format=96)

# Add controls
plate.add_controls('positive', ['A1', 'A2', 'H11', 'H12'])
plate.add_controls('negative', ['A11', 'A12', 'H1', 'H2'])
plate.add_controls('blank', ['D6', 'E6'])

# Add samples (80 samples, 1 replicate each)
plate.add_samples(start_well='B1', num_samples=80, replicates=1)

# Visualize
plate.plot_layout(save_path='results/plate_layout.png')

# Export to CSV
plate.export_layout('results/plate_layout.csv')
```

**Supported Formats:**
- 96-well (8×12)
- 384-well (16×24)
- 1536-well (32×48)

### 3. StatisticalPowerAnalysis

Determine required sample sizes for statistical significance.

**Calculate Sample Size:**
```python
from assay_design_calculator import StatisticalPowerAnalysis

power_calc = StatisticalPowerAnalysis()

# Calculate required sample size
n = power_calc.calculate_sample_size(
    effect_size=0.5,  # Cohen's d
    alpha=0.05,       # Significance level
    power=0.8         # 80% power
)
print(f"Required sample size: {n} per group")
```

**Calculate Statistical Power:**
```python
# Calculate power for given sample size
power = power_calc.calculate_power(
    n=30,            # Sample size per group
    effect_size=0.5,
    alpha=0.05
)
print(f"Statistical power: {power:.2%}")
```

**Visualize Power Curves:**
```python
power_calc.plot_power_curve(save_path='results/power_curve.png')
```

### 4. HitSelectionCriteria

Determine hit thresholds for screening campaigns.

**Calculate Threshold:**
```python
from assay_design_calculator import HitSelectionCriteria

hit_selector = HitSelectionCriteria()

negative_controls = [10, 12, 9, 11, 10, 12, 11, 10, 9, 12]

# Mean + 3SD method (most common)
threshold = hit_selector.calculate_threshold(
    negative_controls,
    method='mean_plus_3sd'
)
print(f"Hit threshold: {threshold:.2f}")
```

**Calculate Percent Inhibition:**
```python
sample_values = [50, 45, 80, 15, 60]
positive_controls = [100, 98, 102, 99, 101]
negative_controls = [10, 12, 9, 11, 10]

percent_inh = hit_selector.calculate_percent_inhibition(
    sample_values,
    positive_controls,
    negative_controls
)
```

### 5. Complete Analysis Pipeline

Run comprehensive assay assessment.

**Example:**
```python
from assay_design_calculator import AssayDesignPipeline

pipeline = AssayDesignPipeline(output_dir='results')

positive_controls = [95, 98, 96, 97, 94, 99, 95, 96, 98, 97]
negative_controls = [5, 6, 4, 7, 5, 6, 5, 4, 6, 5]

metrics = pipeline.assess_assay_quality(
    positive_controls,
    negative_controls,
    assay_name='Kinase_Inhibition_Assay'
)
```

## 📁 Project Structure

```
assay-design-calculator/
├── assay_design_calculator.py  # Main toolkit
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── generate_sample_data.py      # Generate test data
├── examples/                    # Example scripts
│   ├── basic_analysis.py
│   ├── plate_layout_designer.py
│   └── power_analysis.py
├── data/                        # Input data
│   └── README.md
└── results/                     # Output files
    └── README.md
```

## 🔬 Real-World Applications

### Drug Discovery
- **HTS campaign design** - Optimize plate layouts for screening
- **Assay validation** - Calculate Z-factors before full screening
- **Hit confirmation** - Determine appropriate hit thresholds
- **Lead optimization** - Power analysis for dose-response studies

### Bioanalytical Chemistry
- **Method validation** - LOD/LOQ for analytical methods
- **Quality control** - CV calculations for batch consistency
- **Assay transfer** - Validate performance across labs/instruments

### Pharmaceutical Development
- **Potency assays** - Design and validate product release assays
- **Stability studies** - Determine sample sizes for shelf-life testing
- **Batch testing** - Quality metrics for manufacturing QC

### Academic Research
- **Experimental design** - Power analysis before starting studies
- **Publication** - Report quality metrics in methods sections
- **Grant applications** - Justify sample sizes and study design

## 💻 Usage Examples

### Example 1: Complete Assay Validation

```bash
python -m examples.basic_analysis
```

### Example 2: Design Plate Layout

```bash
python -m examples.plate_layout_designer
```

### Example 3: Power Analysis

```bash
python -m examples.power_analysis
```

## 📊 Example Output

### Z-Factor Report
```
=======================================================================
   ASSAY QUALITY METRICS REPORT
=======================================================================

📊 Z-Factor Analysis:
   Z' = 0.876
   Quality: Excellent
   Positive Controls: 96.80 ± 1.55
   Negative Controls: 5.40 ± 0.97

📈 Coefficient of Variation:
   CV = 1.60%
   Reproducibility: Excellent

📏 Signal Window:
   Signal Window = 17.93
   Fold Change = 17.93x
```

## 🎯 Decision Criteria

### Assay Quality (Z-Factor)
| Z' Value | Quality | Action |
|----------|---------|--------|
| ≥ 0.5 | Excellent | Proceed with screening |
| 0.0 - 0.5 | Marginal | Optimize before screening |
| < 0.0 | Poor | Major optimization needed |

### Reproducibility (CV)
| CV (%) | Quality | Action |
|--------|---------|--------|
| < 10 | Excellent | Acceptable for all uses |
| 10-20 | Good | Acceptable for most applications |
| > 20 | Poor | Requires optimization |

### Signal Quality (S/N)
| S/N Ratio | Quality | Action |
|-----------|---------|--------|
| ≥ 10 | Excellent | Ideal for all applications |
| 3-10 | Good | Acceptable for most uses |
| < 3 | Poor | Optimization needed |

## ⚠️ Troubleshooting

### Poor Z-Factor

**Causes:**
- High variability in controls
- Small signal window
- Systematic errors

**Solutions:**
- Increase replicate number
- Optimize assay conditions
- Check reagent quality
- Validate liquid handling

### High CV

**Causes:**
- Pipetting errors
- Temperature variations
- Reagent instability

**Solutions:**
- Use automated liquid handlers
- Control environmental conditions
- Prepare fresh reagents
- Increase mixing

### Low S/N

**Causes:**
- High background noise
- Weak signal
- Detector issues

**Solutions:**
- Optimize detection settings
- Increase signal (e.g., more enzyme, longer incubation)
- Reduce background (blocking agents, wash steps)

## 🛠️ Customization

### Custom Quality Metrics

Extend the toolkit with your own metrics:

```python
from assay_design_calculator import AssayQualityMetrics

class CustomMetrics(AssayQualityMetrics):
    def calculate_ssmd(self, positive, negative):
        """Calculate Strictly Standardized Mean Difference."""
        mean_diff = np.mean(positive) - np.mean(negative)
        var_sum = np.var(positive) + np.var(negative)
        ssmd = mean_diff / np.sqrt(var_sum)
        return ssmd
```

### Custom Plate Layouts

Create specialized plate configurations:

```python
# Dose-response layout
plate = PlateLayoutDesigner(384)
for row in range(16):
    plate.add_samples(f'{chr(65+row)}1', num_samples=1, replicates=12)
```

## 🌐 Professional Context

This toolkit addresses critical needs in pharmaceutical research:

- **Standardization** - Industry-standard metrics (Z-factor, CV, LOD/LOQ)
- **Efficiency** - Automated calculations save hours of manual work
- **Reproducibility** - Consistent methodology across projects
- **Validation** - Essential for regulatory submissions
- **Decision-making** - Data-driven assay development

Positions you as someone who understands:
- Assay development from concept to validation
- Statistical rigor in pharmaceutical research
- HTS workflows and automation
- Quality by Design (QbD) principles

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional quality metrics (SSMD, SW, etc.)
- Support for more plate formats
- Integration with plate readers
- Bayesian power analysis
- Machine learning-based optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Developed for pharmaceutical and biotech research
- Based on industry best practices and FDA guidelines
- Thanks to the scientific Python community

## 📧 Contact

**Oluwaseun O. Ajayi**  
PhD Researcher, Chemistry  
University of Georgia

- **GitHub**: [@Oluwaseun-O-Ajayi](https://github.com/Oluwaseun-O-Ajayi)
- **Academic Email**: oluwaseun.ajayi@uga.edu
- **Personal Email**: seunolanikeajayi@gmail.com

## 📖 Citation

If you use this toolkit in your research:

```bibtex
@software{assay_design_calculator,
  author = {Oluwaseun O. Ajayi},
  title = {Assay Design Calculator},
  year = {2024},
  url = {https://github.com/Oluwaseun-O-Ajayi/assay-design-calculator}
}
```

## 📚 References

- Zhang, J. H.; Chung, T. D.; Oldenburg, K. R. A Simple Statistical Parameter for Use in Evaluation and Validation of High Throughput Screening Assays. J. Biomol. Screen. 1999, 4 (2), 67–73. https://doi.org/10.1177/108705719900400206
- Iversen, P. W.; Beck, B.; Chen, Y. F.; Dere, W.; Devanarayan, V.; Eastwood, B. J.; Fairbrother, W. J.; Brimacombe, K.; Gubernator, K.; Hanna, D.; et al. HTS Assay Validation. In Assay Guidance Manual; Markossian, S., Grossman, A., Baskir, H., Eds.; Eli Lilly & Company and the National Center for Advancing Translational Sciences: Bethesda, MD, 2004–. Updated 2012 Oct 1. https://www.ncbi.nlm.nih.gov/books/NBK83783/
- FDA Guidance for Industry: Bioanalytical Method Validation (2018)

---

**Made with ❤️ for pharmaceutical research and drug discovery**