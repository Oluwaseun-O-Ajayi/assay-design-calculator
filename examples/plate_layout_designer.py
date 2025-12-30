"""
Plate Layout Design Example

Demonstrates how to design and visualize microplate layouts.
"""

import sys
sys.path.append('..')

from assay_design_calculator import PlateLayoutDesigner

print("""
╔═══════════════════════════════════════════════════════════════════╗
║     PLATE LAYOUT DESIGNER                                         ║
╚═══════════════════════════════════════════════════════════════════╝
""")

# Example 1: Standard 96-well plate
print("\n=== Example 1: Standard 96-Well Plate ===\n")

plate96 = PlateLayoutDesigner(plate_format=96)

# Add controls in corners
plate96.add_controls('positive', ['A1', 'A2', 'H11', 'H12'])
plate96.add_controls('negative', ['A11', 'A12', 'H1', 'H2'])
plate96.add_controls('blank', ['D6', 'E6'])

# Add 80 samples with single replicates
plate96.add_samples(start_well='B1', num_samples=80, replicates=1)

print("96-well plate designed:")
print(f"  Positive controls: 4 wells")
print(f"  Negative controls: 4 wells")
print(f"  Blank controls: 2 wells")
print(f"  Samples: 80 wells")

# Visualize
plate96.plot_layout(save_path='results/plate_96well.png')

# Export to CSV
plate96.export_layout('results/plate_96well_layout.csv')

# Example 2: 384-well plate with replicates
print("\n\n=== Example 2: 384-Well Plate with Duplicates ===\n")

plate384 = PlateLayoutDesigner(plate_format=384)

# Add controls in corners (more replicates for 384-well)
positive_wells = [f'A{i}' for i in range(1, 9)] + [f'P{i}' for i in range(17, 25)]
negative_wells = [f'A{i}' for i in range(17, 25)] + [f'P{i}' for i in range(1, 9)]

plate384.add_controls('positive', positive_wells)
plate384.add_controls('negative', negative_wells)

# Add samples with duplicates
plate384.add_samples(start_well='B1', num_samples=150, replicates=2)

print("384-well plate designed:")
print(f"  Positive controls: {len(positive_wells)} wells")
print(f"  Negative controls: {len(negative_wells)} wells")
print(f"  Samples: 150 compounds × 2 replicates = 300 wells")

# Visualize
plate384.plot_layout(save_path='results/plate_384well.png')

# Example 3: Dose-response plate
print("\n\n=== Example 3: Dose-Response Plate (96-well) ===\n")

plate_dr = PlateLayoutDesigner(plate_format=96)

# Controls
plate_dr.add_controls('positive', ['A1', 'A2'])
plate_dr.add_controls('negative', ['H11', 'H12'])

# Each compound in one row (12 concentrations per row)
print("Dose-response design:")
print("  Each row = 1 compound tested at 10 dose points")
print("  Rows B-G = 6 compounds")

# Manually design dose-response
# (In practice, you'd add a special method for this)
for row_idx, row_letter in enumerate(['B', 'C', 'D', 'E', 'F', 'G']):
    for col in range(1, 11):  # 10 dose points
        well = f'{row_letter}{col}'
        # This would normally be done with a custom method
        # For now, we'll add them as samples
        
plate_dr.add_samples(start_well='B1', num_samples=60, replicates=1)

plate_dr.plot_layout(save_path='results/plate_dose_response.png')

print("\n" + "="*70)
print("✅ Plate Layouts Complete!")
print("📁 Check 'results/' folder for:")
print("   - plate_96well.png")
print("   - plate_96well_layout.csv")
print("   - plate_384well.png")
print("   - plate_dose_response.png")
print("="*70 + "\n")