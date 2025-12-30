# Results Directory

Analysis outputs will be saved here.

## Generated Files

When you run the example scripts, the following files will be created:

### Quality Analysis
- `Excellent_Assay_quality.png` - Control comparison plots
- `Marginal_Assay_quality.png` - Control comparison plots

### Plate Layouts
- `plate_96well.png` - 96-well plate visualization
- `plate_96well_layout.csv` - Plate layout data
- `plate_384well.png` - 384-well plate visualization
- `plate_dose_response.png` - Dose-response plate layout

### Power Analysis
- `power_curve.png` - Sample size vs. effect size curves

## Running Examples

Generate example data first:
```bash
python generate_sample_data.py
```

Then run analyses:
```bash
# Basic quality metrics
python -m examples.basic_analysis

# Plate layout design
python -m examples.plate_layout_designer

# Power analysis
python -m examples.power_analysis
```

## Note

Generated output files (PNG, CSV) are not tracked by git to keep the repository size small. See `.gitignore`.