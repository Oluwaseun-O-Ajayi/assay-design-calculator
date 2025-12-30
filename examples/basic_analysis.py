"""
Basic Assay Quality Analysis Example

Demonstrates basic usage of the Assay Design Calculator.
"""

import sys
sys.path.append('..')

import pandas as pd
from assay_design_calculator import AssayDesignPipeline, AssayQualityMetrics

print("""
╔═══════════════════════════════════════════════════════════════════╗
║     BASIC ASSAY QUALITY ANALYSIS                                  ║
╚═══════════════════════════════════════════════════════════════════╝
""")

# Example 1: Analyze excellent assay
print("\n=== Example 1: Excellent Assay Analysis ===\n")

try:
    # Load data
    df = pd.read_csv('data/excellent_assay.csv')
    
    # Separate controls
    positive = df[df['type'] == 'positive']['value'].values
    negative = df[df['type'] == 'negative']['value'].values
    
    # Create pipeline
    pipeline = AssayDesignPipeline(output_dir='results')
    
    # Run analysis
    metrics = pipeline.assess_assay_quality(
        positive,
        negative,
        assay_name='Excellent_Assay'
    )
    
except FileNotFoundError:
    print("⚠️  Data file not found!")
    print("Run 'python generate_sample_data.py' first to create example data.")
    sys.exit(1)

# Example 2: Analyze marginal assay
print("\n\n=== Example 2: Marginal Assay Analysis ===\n")

try:
    # Load data
    df2 = pd.read_csv('data/marginal_assay.csv')
    
    # Separate controls
    positive2 = df2[df2['type'] == 'positive']['value'].values
    negative2 = df2[df2['type'] == 'negative']['value'].values
    
    # Run analysis
    metrics2 = pipeline.assess_assay_quality(
        positive2,
        negative2,
        assay_name='Marginal_Assay'
    )
    
except FileNotFoundError:
    print("⚠️  Data file not found!")

# Example 3: Compare multiple metrics
print("\n\n=== Example 3: Detailed Metrics Comparison ===\n")

qc = AssayQualityMetrics()

# Calculate all metrics for excellent assay
print("Excellent Assay:")
z = qc.calculate_z_factor(positive, negative)
cv_pos = qc.calculate_cv(positive)
cv_neg = qc.calculate_cv(negative)
sw = qc.calculate_signal_window(positive, negative)
lod_loq = qc.calculate_detection_limits(negative)

print(f"  Z-factor: {z:.3f}")
print(f"  CV (positive): {cv_pos:.2f}%")
print(f"  CV (negative): {cv_neg:.2f}%")
print(f"  Signal Window: {sw:.2f}x")
print(f"  LOD: {lod_loq['LOD']:.2f} {lod_loq['units']}")

print("\n" + "="*70)
print("✅ Analysis Complete!")
print("📁 Check 'results/' folder for:")
print("   - Excellent_Assay_quality.png")
print("   - Marginal_Assay_quality.png")
print("="*70 + "\n")