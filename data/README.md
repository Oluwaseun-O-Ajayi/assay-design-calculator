# Data Directory

Place your assay data files here for analysis.

## Generating Example Data

Run the data generator to create sample datasets:

```bash
python generate_sample_data.py
```

This creates:
- `excellent_assay.csv` - High Z-factor assay
- `marginal_assay.csv` - Low Z-factor assay
- `screening_data.csv` - HTS screening with hits
- `dose_response.csv` - IC50 dose-response data

## Data Format

### Control Data

CSV with columns: `well`, `type`, `value`

```csv
well,type,value
A1,positive,98.5
A2,positive,97.2
H11,negative,10.3
H12,negative,9.8
```

### Screening Data

CSV with columns: `well`, `compound_id`, `value`

```csv
well,compound_id,value
A1,CPD_0001,12.5
A2,CPD_0002,85.3
A3,CPD_0003,11.2
```

### Dose-Response Data

CSV with columns: `concentration_uM`, `replicate`, `percent_inhibition`

```csv
concentration_uM,replicate,percent_inhibition
0.01,1,5.2
0.01,2,6.1
0.1,1,10.5
0.1,2,11.2
```

## Note

Data files are not tracked by git. See `.gitignore`.