# Replication Code: Macroeconomic Shocks and Effective Dimension

This repository contains all code necessary to replicate the empirical analysis in 
"Macroeconomic Shocks, Arbitrage Constraints, and the Time-Varying Effective Dimension 
of Asset Returns."

## Directory Structure

```
replication/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config/
│   └── settings.py            # Global configuration and parameters
├── data/
│   ├── raw/                   # Downloaded raw data (auto-populated)
│   ├── processed/             # Cleaned and merged data
│   └── manual/                # DATA YOU MUST PROVIDE (see below)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py        # Download publicly available data
│   │   ├── process.py         # Clean and merge all data
│   │   └── load.py            # Load processed data for analysis
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── effective_dimension.py  # Compute time-varying dimension
│   │   ├── shock_response.py       # Test H1: shock → dimension
│   │   ├── constraint_interaction.py # Test H2: shock × constraint
│   │   ├── portfolio_loadings.py   # Test H3: portfolio loadings
│   │   └── robustness.py           # All robustness checks
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Statistical and utility functions
├── output/
│   ├── tables/                # LaTeX tables
│   └── figures/               # PDF/PNG figures
├── tests/
│   └── test_dimension.py      # Unit tests
├── run_all.py                 # Master script to run everything
└── run_analysis.py            # Run analysis only (assumes data ready)
```

## ⚠️ MANUAL DATA REQUIREMENTS

The following data series are **NOT publicly downloadable** and must be provided manually.
Place these files in `data/manual/` before running the code.

### 1. Monetary Policy Shocks (REQUIRED)
**File:** `data/manual/monetary_shocks.csv`
**Source:** Nakamura & Steinsson (2018), contact authors or construct from futures data
**Format:**
```csv
date,shock
1990-01-31,-0.05
1990-02-28,0.12
...
```
- `date`: End of month (YYYY-MM-DD)
- `shock`: Monetary policy surprise in percentage points (can be positive or negative)

### 2. Fiscal Policy Shocks (OPTIONAL - for robustness)
**File:** `data/manual/fiscal_shocks.csv`
**Source:** Ramey & Zubairy (2018), available from Valerie Ramey's website
**Format:**
```csv
date,shock
1990-03-31,0.002
1990-06-30,-0.001
...
```
- `date`: End of quarter (YYYY-MM-DD)  
- `shock`: Defense news shock as fraction of GDP

### 3. Intermediary Capital Ratio (OPTIONAL - we attempt download)
**File:** `data/manual/intermediary_capital.csv`
**Source:** He, Kelly & Manela (2017), from Asaf Manela's website
**Note:** Code attempts to download this automatically. Only provide manually if download fails.
**Format:**
```csv
date,capital_ratio
1990-03-31,0.052
1990-06-30,0.048
...
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place manual data files
Copy your manually obtained data to `data/manual/`:
- `monetary_shocks.csv` (REQUIRED)
- `fiscal_shocks.csv` (optional)

### 3. Run everything
```bash
python run_all.py
```

Or run steps individually:
```bash
python -m src.data.download      # Download public data
python -m src.data.process       # Process and merge
python run_analysis.py           # Run all analysis
```

### 4. Find outputs
- Tables: `output/tables/` (LaTeX format)
- Figures: `output/figures/` (PDF format)

## Configuration

Edit `config/settings.py` to modify:
- Sample period (default: 1990-2024)
- Rolling window length (default: 60 months)
- Shock threshold percentile (default: 10th)
- Constraint threshold percentile (default: 75th)

## Outputs Produced

### Tables
| File | Description | Paper Reference |
|------|-------------|-----------------|
| `table1_data_summary.tex` | Data sources and coverage | Table 1 |
| `table2_dimension_shocks.tex` | Dimension response to shocks | Table 2 |
| `table3_loadings.tex` | Portfolio loadings on priced directions | Table 3 |
| `table4_state_returns.tex` | Returns conditional on dimension | Table 4 |
| `table5_shock_constraint.tex` | Effect of shocks on constraints | Table 5 |
| `table6_interaction.tex` | Shock × constraint interaction | Table 6 |
| `table7_placebo.tex` | Placebo tests | Table 7 |
| `tableA1_robustness_window.tex` | Robustness to window length | Table A1 |
| `tableA2_robustness_constraint.tex` | Robustness to constraint spec | Table A2 |

### Figures
| File | Description | Paper Reference |
|------|-------------|-----------------|
| `figure1_dimension_diagram.pdf` | Conceptual diagram | Figure 1 |
| `figure2_dimension_timeseries.pdf` | Effective dimension over time | Figure 2 |
| `figure3_eigenvalue_structure.pdf` | Cumulative variance share | Figure 3 |
| `figure4_scree_comparison.pdf` | Scree plots: normal vs crisis | Figure 4 |

## Runtime

Expected runtime on a standard laptop:
- Data download: 2-5 minutes
- Data processing: 1-2 minutes
- Full analysis: 5-10 minutes
- **Total: ~15 minutes**

## Requirements

- Python 3.9+
- See `requirements.txt` for packages

## Contact

For questions about the code, contact [author].
For questions about manual data sources, see the original papers cited above.
