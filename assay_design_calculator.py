"""
Assay Design Calculator
=======================

A comprehensive Python toolkit for pharmaceutical assay design and validation.
Designed for high-throughput screening (HTS), drug discovery, and bioanalytical
research.

Features:
- Z-factor and Z' calculation for assay quality assessment
- Signal-to-noise ratio (S/N) analysis
- Coefficient of variation (CV) calculations
- Limit of detection (LOD) and quantification (LOQ)
- Plate layout design and optimization
- Statistical power analysis for sample size determination
- Hit selection criteria and threshold determination

Author: Oluwaseun O. Ajayi
Institution: University of Georgia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class AssayQualityMetrics:
    """
    Calculate industry-standard assay quality metrics.
    """
    
    def __init__(self):
        """Initialize quality metrics calculator."""
        self.metrics = {}
    
    def calculate_z_factor(self, positive_controls, negative_controls):
        """
        Calculate Z-factor (Z') for assay quality assessment.
        
        Z' = 1 - (3 × (σp + σn)) / |μp - μn|
        
        Interpretation:
        - Z' > 0.5: Excellent assay
        - 0.5 > Z' > 0: Marginal assay
        - Z' < 0: Poor assay (unacceptable)
        
        Args:
            positive_controls: Array of positive control values
            negative_controls: Array of negative control values
        
        Returns:
            Z-factor value
        """
        pos = np.array(positive_controls)
        neg = np.array(negative_controls)
        
        # Calculate means and standard deviations
        mean_pos = np.mean(pos)
        mean_neg = np.mean(neg)
        std_pos = np.std(pos, ddof=1)
        std_neg = np.std(neg, ddof=1)
        
        # Calculate Z-factor
        z_factor = 1 - (3 * (std_pos + std_neg)) / abs(mean_pos - mean_neg)
        
        # Store metrics
        self.metrics['z_factor'] = {
            'value': z_factor,
            'pos_mean': mean_pos,
            'pos_std': std_pos,
            'neg_mean': mean_neg,
            'neg_std': std_neg,
            'quality': self._assess_z_factor(z_factor)
        }
        
        return z_factor
    
    def _assess_z_factor(self, z_factor):
        """Assess assay quality based on Z-factor."""
        if z_factor >= 0.5:
            return "Excellent"
        elif z_factor >= 0:
            return "Marginal"
        else:
            return "Poor (Unacceptable)"
    
    def calculate_signal_to_noise(self, signal, noise=None, blank=None):
        """
        Calculate signal-to-noise ratio (S/N).
        
        S/N = (Signal - Blank) / Noise
        
        If noise is not provided, it's calculated as the standard deviation
        of the blank measurements.
        
        Args:
            signal: Signal measurement(s)
            noise: Noise measurement (optional)
            blank: Blank measurements (required if noise not provided)
        
        Returns:
            S/N ratio
        """
        signal = np.array(signal)
        
        if noise is None:
            if blank is None:
                raise ValueError("Either noise or blank measurements must be provided")
            blank = np.array(blank)
            noise = np.std(blank, ddof=1)
            mean_blank = np.mean(blank)
        else:
            mean_blank = 0 if blank is None else np.mean(blank)
        
        # Calculate S/N
        sn_ratio = (signal - mean_blank) / noise
        
        # Store metrics
        self.metrics['signal_to_noise'] = {
            'value': np.mean(sn_ratio) if isinstance(sn_ratio, np.ndarray) else sn_ratio,
            'noise': noise,
            'quality': "Good" if np.mean(sn_ratio) >= 3 else "Poor"
        }
        
        return sn_ratio
    
    def calculate_cv(self, data, as_percentage=True):
        """
        Calculate coefficient of variation (CV).
        
        CV = (σ / μ) × 100%
        
        Interpretation:
        - CV < 10%: Excellent reproducibility
        - 10% ≤ CV < 20%: Good reproducibility
        - CV ≥ 20%: Poor reproducibility
        
        Args:
            data: Array of measurements
            as_percentage: Return as percentage (default: True)
        
        Returns:
            CV value
        """
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        cv = (std / mean) * (100 if as_percentage else 1)
        
        # Store metrics
        self.metrics['cv'] = {
            'value': cv,
            'mean': mean,
            'std': std,
            'quality': self._assess_cv(cv)
        }
        
        return cv
    
    def _assess_cv(self, cv):
        """Assess reproducibility based on CV."""
        if cv < 10:
            return "Excellent"
        elif cv < 20:
            return "Good"
        else:
            return "Poor"
    
    def calculate_detection_limits(self, blank_measurements, slope=None):
        """
        Calculate Limit of Detection (LOD) and Limit of Quantification (LOQ).
        
        LOD = 3.3 × (σ / S)
        LOQ = 10 × (σ / S)
        
        where σ is the standard deviation of blank and S is the slope of
        the calibration curve.
        
        Args:
            blank_measurements: Array of blank measurements
            slope: Slope of calibration curve (optional, returns relative values if not provided)
        
        Returns:
            Dictionary with LOD and LOQ
        """
        blank = np.array(blank_measurements)
        std_blank = np.std(blank, ddof=1)
        
        if slope is None:
            # Return relative LOD/LOQ (in signal units)
            lod = 3.3 * std_blank
            loq = 10 * std_blank
            units = "signal units"
        else:
            # Return absolute LOD/LOQ (in concentration units)
            lod = (3.3 * std_blank) / slope
            loq = (10 * std_blank) / slope
            units = "concentration units"
        
        # Store metrics
        self.metrics['detection_limits'] = {
            'lod': lod,
            'loq': loq,
            'units': units,
            'blank_std': std_blank
        }
        
        return {'LOD': lod, 'LOQ': loq, 'units': units}
    
    def calculate_signal_window(self, positive_controls, negative_controls):
        """
        Calculate signal window (dynamic range).
        
        Signal Window = μ_pos / μ_neg
        
        Args:
            positive_controls: Array of positive control values
            negative_controls: Array of negative control values
        
        Returns:
            Signal window value
        """
        mean_pos = np.mean(positive_controls)
        mean_neg = np.mean(negative_controls)
        
        signal_window = mean_pos / mean_neg
        
        self.metrics['signal_window'] = {
            'value': signal_window,
            'fold_change': signal_window
        }
        
        return signal_window
    
    def generate_quality_report(self):
        """Generate comprehensive quality metrics report."""
        print("\n" + "="*70)
        print("   ASSAY QUALITY METRICS REPORT")
        print("="*70)
        
        if 'z_factor' in self.metrics:
            z = self.metrics['z_factor']
            print(f"\n📊 Z-Factor Analysis:")
            print(f"   Z' = {z['value']:.3f}")
            print(f"   Quality: {z['quality']}")
            print(f"   Positive Controls: {z['pos_mean']:.2f} ± {z['pos_std']:.2f}")
            print(f"   Negative Controls: {z['neg_mean']:.2f} ± {z['neg_std']:.2f}")
        
        if 'signal_to_noise' in self.metrics:
            sn = self.metrics['signal_to_noise']
            print(f"\n🔊 Signal-to-Noise Ratio:")
            print(f"   S/N = {sn['value']:.2f}")
            print(f"   Noise = {sn['noise']:.2f}")
            print(f"   Quality: {sn['quality']}")
        
        if 'cv' in self.metrics:
            cv = self.metrics['cv']
            print(f"\n📈 Coefficient of Variation:")
            print(f"   CV = {cv['value']:.2f}%")
            print(f"   Mean = {cv['mean']:.2f}")
            print(f"   Std Dev = {cv['std']:.2f}")
            print(f"   Reproducibility: {cv['quality']}")
        
        if 'detection_limits' in self.metrics:
            dl = self.metrics['detection_limits']
            print(f"\n🔍 Detection Limits:")
            print(f"   LOD = {dl['lod']:.4f} {dl['units']}")
            print(f"   LOQ = {dl['loq']:.4f} {dl['units']}")
        
        if 'signal_window' in self.metrics:
            sw = self.metrics['signal_window']
            print(f"\n📏 Signal Window:")
            print(f"   Signal Window = {sw['value']:.2f}")
            print(f"   Fold Change = {sw['fold_change']:.2f}x")
        
        print("\n" + "="*70)
        
        return self.metrics


class PlateLayoutDesigner:
    """
    Design and visualize microplate layouts for HTS assays.
    """
    
    def __init__(self, plate_format=96):
        """
        Initialize plate layout designer.
        
        Args:
            plate_format: 96, 384, or 1536 well plate
        """
        self.plate_format = plate_format
        self.rows, self.cols = self._get_plate_dimensions(plate_format)
        self.layout = np.empty((self.rows, self.cols), dtype=object)
        self.layout[:] = 'Empty'
    
    def _get_plate_dimensions(self, plate_format):
        """Get plate dimensions based on format."""
        dimensions = {
            96: (8, 12),
            384: (16, 24),
            1536: (32, 48)
        }
        if plate_format not in dimensions:
            raise ValueError(f"Plate format {plate_format} not supported. Use 96, 384, or 1536.")
        return dimensions[plate_format]
    
    def add_controls(self, control_type, wells):
        """
        Add control wells to the layout.
        
        Args:
            control_type: 'positive', 'negative', or 'blank'
            wells: List of well positions (e.g., ['A1', 'A2', 'H11', 'H12'])
        """
        for well in wells:
            row, col = self._parse_well_position(well)
            self.layout[row, col] = control_type.capitalize()
    
    def add_samples(self, start_well, num_samples, replicates=1):
        """
        Add sample wells to the layout.
        
        Args:
            start_well: Starting well position (e.g., 'B1')
            num_samples: Number of unique samples
            replicates: Number of replicates per sample
        """
        row, col = self._parse_well_position(start_well)
        sample_num = 1
        
        for _ in range(num_samples):
            for _ in range(replicates):
                if row < self.rows and col < self.cols:
                    if self.layout[row, col] == 'Empty':
                        self.layout[row, col] = f'Sample_{sample_num}'
                    
                    # Move to next well
                    col += 1
                    if col >= self.cols:
                        col = 0
                        row += 1
                        if row >= self.rows:
                            print(f"⚠️ Warning: Ran out of wells after {sample_num} samples")
                            return
            sample_num += 1
    
    def _parse_well_position(self, well):
        """Convert well position (e.g., 'A1') to row, col indices."""
        row_letter = well[0].upper()
        col_number = int(well[1:])
        
        row = ord(row_letter) - ord('A')
        col = col_number - 1
        
        return row, col
    
    def plot_layout(self, save_path=None):
        """Visualize the plate layout."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create color map
        unique_types = np.unique(self.layout)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        color_map = {well_type: colors[i] for i, well_type in enumerate(unique_types)}
        
        # Create colored layout
        colored_layout = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                colored_layout[i, j] = color_map[self.layout[i, j]][:3]
        
        # Plot
        ax.imshow(colored_layout, aspect='auto')
        
        # Add grid
        ax.set_xticks(np.arange(self.cols))
        ax.set_yticks(np.arange(self.rows))
        ax.set_xticklabels(range(1, self.cols + 1))
        ax.set_yticklabels([chr(65 + i) for i in range(self.rows)])
        ax.grid(which='both', color='black', linewidth=0.5)
        
        # Add well labels
        for i in range(self.rows):
            for j in range(self.cols):
                well_type = self.layout[i, j]
                if well_type != 'Empty':
                    # Shorten sample names for display
                    display_text = well_type.replace('Sample_', 'S')
                    ax.text(j, i, display_text, ha='center', va='center',
                           fontsize=6 if self.plate_format == 384 else 8,
                           fontweight='bold')
        
        ax.set_title(f'{self.plate_format}-Well Plate Layout', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Column', fontsize=12, fontweight='bold')
        ax.set_ylabel('Row', fontsize=12, fontweight='bold')
        
        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color_map[wt][:3], 
                                        edgecolor='black', label=wt) 
                          for wt in unique_types]
        ax.legend(handles=legend_elements, loc='center left', 
                 bbox_to_anchor=(1, 0.5), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plate layout saved to: {save_path}")
        
        plt.show()
    
    def export_layout(self, filename):
        """Export plate layout to CSV."""
        df = pd.DataFrame(self.layout)
        df.columns = [str(i+1) for i in range(self.cols)]
        df.index = [chr(65 + i) for i in range(self.rows)]
        df.to_csv(filename)
        print(f"✅ Plate layout exported to: {filename}")


class StatisticalPowerAnalysis:
    """
    Perform statistical power analysis for sample size determination.
    """
    
    def __init__(self):
        """Initialize power analysis calculator."""
        pass
    
    def calculate_sample_size(self, effect_size, alpha=0.05, power=0.8, alternative='two-sided'):
        """
        Calculate required sample size for detecting an effect.
        
        Args:
            effect_size: Cohen's d (standardized effect size)
            alpha: Significance level (default: 0.05)
            power: Statistical power (default: 0.8 = 80%)
            alternative: 'two-sided' or 'one-sided'
        
        Returns:
            Required sample size per group
        """
        # Z-scores for alpha and power
        if alternative == 'two-sided':
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Calculate sample size
        n = ((z_alpha + z_beta) / effect_size) ** 2
        
        # Round up
        n = int(np.ceil(n))
        
        return n
    
    def calculate_power(self, n, effect_size, alpha=0.05, alternative='two-sided'):
        """
        Calculate statistical power given sample size.
        
        Args:
            n: Sample size per group
            effect_size: Cohen's d
            alpha: Significance level
            alternative: 'two-sided' or 'one-sided'
        
        Returns:
            Statistical power
        """
        if alternative == 'two-sided':
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        # Calculate power
        z_beta = effect_size * np.sqrt(n) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return power
    
    def plot_power_curve(self, effect_sizes=None, alpha=0.05, power=0.8, save_path=None):
        """Plot sample size vs. effect size for different power levels."""
        if effect_sizes is None:
            effect_sizes = np.linspace(0.1, 2.0, 100)
        
        power_levels = [0.7, 0.8, 0.9, 0.95]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for pwr in power_levels:
            sample_sizes = [self.calculate_sample_size(es, alpha, pwr) 
                           for es in effect_sizes]
            ax.plot(effect_sizes, sample_sizes, linewidth=2, 
                   label=f'Power = {pwr}')
        
        ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Size (per group)', fontsize=12, fontweight='bold')
        ax.set_title('Statistical Power Analysis: Sample Size Requirements', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Power curve saved to: {save_path}")
        
        plt.show()


class HitSelectionCriteria:
    """
    Determine hit selection thresholds for screening assays.
    """
    
    def __init__(self):
        """Initialize hit selection calculator."""
        pass
    
    def calculate_threshold(self, negative_controls, method='mean_plus_3sd'):
        """
        Calculate hit threshold based on negative controls.
        
        Methods:
        - 'mean_plus_3sd': Mean + 3×SD
        - 'mean_plus_2sd': Mean + 2×SD
        - 'percentile_99': 99th percentile
        - 'percentile_95': 95th percentile
        
        Args:
            negative_controls: Array of negative control values
            method: Threshold calculation method
        
        Returns:
            Threshold value
        """
        neg = np.array(negative_controls)
        mean = np.mean(neg)
        std = np.std(neg, ddof=1)
        
        if method == 'mean_plus_3sd':
            threshold = mean + 3 * std
        elif method == 'mean_plus_2sd':
            threshold = mean + 2 * std
        elif method == 'percentile_99':
            threshold = np.percentile(neg, 99)
        elif method == 'percentile_95':
            threshold = np.percentile(neg, 95)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return threshold
    
    def calculate_percent_inhibition(self, sample_values, positive_controls, 
                                    negative_controls):
        """
        Calculate percent inhibition for samples.
        
        % Inhibition = 100 × (1 - (Sample - Neg) / (Pos - Neg))
        
        Args:
            sample_values: Array of sample measurements
            positive_controls: Array of positive control values
            negative_controls: Array of negative control values
        
        Returns:
            Array of percent inhibition values
        """
        samples = np.array(sample_values)
        mean_pos = np.mean(positive_controls)
        mean_neg = np.mean(negative_controls)
        
        percent_inh = 100 * (1 - (samples - mean_neg) / (mean_pos - mean_neg))
        
        return percent_inh


# ============================================================================
# Complete Analysis Pipeline
# ============================================================================

class AssayDesignPipeline:
    """
    Complete pipeline for assay design and quality assessment.
    """
    
    def __init__(self, output_dir='results'):
        """
        Initialize analysis pipeline.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"   ASSAY DESIGN & QUALITY ASSESSMENT PIPELINE")
        print(f"{'='*70}\n")
        print(f"📁 Output directory: {self.output_dir}")
    
    def assess_assay_quality(self, positive_controls, negative_controls, 
                            assay_name='Assay'):
        """
        Complete assay quality assessment.
        
        Args:
            positive_controls: Array of positive control values
            negative_controls: Array of negative control values
            assay_name: Name of the assay
        """
        print(f"\n{'='*70}")
        print(f"Assessing: {assay_name}")
        print(f"{'='*70}")
        
        # Initialize calculator
        qc = AssayQualityMetrics()
        
        # Calculate all metrics
        print("\n📊 Calculating Quality Metrics...")
        
        z_factor = qc.calculate_z_factor(positive_controls, negative_controls)
        signal_window = qc.calculate_signal_window(positive_controls, negative_controls)
        cv_pos = qc.calculate_cv(positive_controls)
        cv_neg = qc.calculate_cv(negative_controls)
        
        # Generate report
        qc.generate_quality_report()
        
        # Create visualization
        self._plot_control_comparison(positive_controls, negative_controls, 
                                     assay_name, z_factor)
        
        return qc.metrics


    def _plot_control_comparison(self, pos, neg, assay_name, z_factor):
        """Plot positive vs negative control comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        data = [pos, neg]
        bp = ax1.boxplot(data, labels=['Positive', 'Negative'],
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax1.set_ylabel('Signal', fontsize=12, fontweight='bold')
        ax1.set_title(f'{assay_name} - Control Comparison', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add Z-factor text
        ax1.text(0.5, 0.95, f"Z' = {z_factor:.3f}", 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                ha='center', va='top')
        
        # Scatter plot
        x_pos = np.ones(len(pos)) + np.random.normal(0, 0.04, len(pos))
        x_neg = np.ones(len(neg)) * 2 + np.random.normal(0, 0.04, len(neg))
        
        ax2.scatter(x_pos, pos, alpha=0.6, s=50, color='green', 
                   edgecolors='black', linewidth=0.5, label='Positive')
        ax2.scatter(x_neg, neg, alpha=0.6, s=50, color='red',
                   edgecolors='black', linewidth=0.5, label='Negative')
        
        ax2.set_xlim(0.5, 2.5)
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(['Positive', 'Negative'])
        ax2.set_ylabel('Signal', fontsize=12, fontweight='bold')
        ax2.set_title('Individual Measurements', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{assay_name}_quality.png", 
                   dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {self.output_dir / f'{assay_name}_quality.png'}")
        plt.show()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     ASSAY DESIGN CALCULATOR                                       ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    This toolkit provides comprehensive assay design and validation:
    
    1. Z-factor and quality metrics
    2. Signal-to-noise ratio analysis
    3. Coefficient of variation
    4. Detection limits (LOD/LOQ)
    5. Plate layout design
    6. Statistical power analysis
    7. Hit selection criteria
    
    Example usage:
    
    # Quality metrics
    qc = AssayQualityMetrics()
    z_factor = qc.calculate_z_factor(positive_controls, negative_controls)
    
    # Plate layout
    plate = PlateLayoutDesigner(plate_format=96)
    plate.add_controls('positive', ['A1', 'A2', 'H11', 'H12'])
    plate.add_samples('B1', num_samples=80, replicates=1)
    plate.plot_layout()
    
    # Power analysis
    power = StatisticalPowerAnalysis()
    n = power.calculate_sample_size(effect_size=0.5, power=0.8)
    """)