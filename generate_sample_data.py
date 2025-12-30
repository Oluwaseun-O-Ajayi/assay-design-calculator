"""
Generate Example Assay Data

Creates synthetic assay data for testing the Assay Design Calculator.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_control_data(mean, std, n_samples=10):
    """Generate control measurements."""
    return np.random.normal(mean, std, n_samples)

def main():
    """Generate example assay data files."""
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    print("Generating example assay data...")
    
    # Example 1: Excellent assay (high Z-factor)
    print("\n1. Generating excellent_assay.csv...")
    positive = generate_control_data(mean=100, std=2, n_samples=12)
    negative = generate_control_data(mean=10, std=2, n_samples=12)
    
    df1 = pd.DataFrame({
        'well': [f'A{i+1}' for i in range(12)] + [f'H{i+1}' for i in range(12)],
        'type': ['positive']*12 + ['negative']*12,
        'value': np.concatenate([positive, negative])
    })
    df1.to_csv(data_dir / 'excellent_assay.csv', index=False)
    print(f"   ✅ Created: {data_dir / 'excellent_assay.csv'}")
    print(f"   Expected Z' ≈ 0.8-0.9 (Excellent)")
    
    # Example 2: Marginal assay (low Z-factor)
    print("\n2. Generating marginal_assay.csv...")
    positive = generate_control_data(mean=100, std=10, n_samples=12)
    negative = generate_control_data(mean=60, std=10, n_samples=12)
    
    df2 = pd.DataFrame({
        'well': [f'A{i+1}' for i in range(12)] + [f'H{i+1}' for i in range(12)],
        'type': ['positive']*12 + ['negative']*12,
        'value': np.concatenate([positive, negative])
    })
    df2.to_csv(data_dir / 'marginal_assay.csv', index=False)
    print(f"   ✅ Created: {data_dir / 'marginal_assay.csv'}")
    print(f"   Expected Z' ≈ 0.1-0.3 (Marginal)")
    
    # Example 3: Screening data with hits
    print("\n3. Generating screening_data.csv...")
    np.random.seed(42)
    
    # Generate 96 samples
    n_samples = 96
    baseline = generate_control_data(mean=10, std=2, n_samples=n_samples)
    
    # Add some hits (20% hit rate)
    n_hits = 20
    hit_indices = np.random.choice(n_samples, n_hits, replace=False)
    for idx in hit_indices:
        baseline[idx] = np.random.normal(80, 5)  # High signal for hits
    
    df3 = pd.DataFrame({
        'well': [f'{chr(65 + i//12)}{i%12 + 1}' for i in range(n_samples)],
        'compound_id': [f'CPD_{i+1:04d}' for i in range(n_samples)],
        'value': baseline
    })
    df3.to_csv(data_dir / 'screening_data.csv', index=False)
    print(f"   ✅ Created: {data_dir / 'screening_data.csv'}")
    print(f"   Contains 96 samples with ~20% hits")
    
    # Example 4: Dose-response data
    print("\n4. Generating dose_response.csv...")
    concentrations = [0.01, 0.1, 1, 10, 100, 1000]  # µM
    ic50 = 10  # µM
    hill_slope = -1
    
    responses = []
    for conc in concentrations:
        # 4PL equation with replicates
        response = 100 / (1 + (conc/ic50)**hill_slope)
        # Add replicates with noise
        reps = np.random.normal(response, 5, 3)
        responses.extend(reps)
    
    df4 = pd.DataFrame({
        'concentration_uM': np.repeat(concentrations, 3),
        'replicate': [1, 2, 3] * len(concentrations),
        'percent_inhibition': responses
    })
    df4.to_csv(data_dir / 'dose_response.csv', index=False)
    print(f"   ✅ Created: {data_dir / 'dose_response.csv'}")
    print(f"   IC50 ≈ 10 µM")
    
    print("\n" + "="*60)
    print("✅ Sample data generation complete!")
    print(f"📁 Files saved in: {data_dir.absolute()}")
    print("="*60)
    print("\nYou can now run:")
    print("  python -m examples.basic_analysis")
    print("  python -m examples.plate_layout_designer")
    print("  python -m examples.power_analysis")

if __name__ == "__main__":
    main()