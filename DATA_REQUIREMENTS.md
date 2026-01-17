# Data Requirements for Replication

## Overview

This document specifies all data needed to replicate the empirical analysis in "Macroeconomic Shocks, Arbitrage Constraints, and the Time-Varying Effective Dimension of Asset Returns."

**Binding Sample Period:** 1990–2024 (determined by monetary policy shock availability)

---

## 1. MACROECONOMIC SHOCKS

### 1.1 Monetary Policy Shocks (PRIMARY)

| Item | Specification |
|------|---------------|
| **Variable** | High-frequency monetary policy surprises |
| **Source** | Nakamura & Steinsson (2018), updated |
| **Original Paper** | "High-Frequency Identification of Monetary Non-Neutrality: The Information Effect," *QJE* |
| **Frequency** | Event-level (FOMC meetings), aggregated to monthly |
| **Sample** | 1990–2024 |
| **Construction** | Change in federal funds futures (FF4) in 30-minute window around FOMC announcements |

**Where to Get It:**
- Authors' websites: https://www.columbia.edu/~js3204/
- Replication files from QJE
- Alternative: Construct from CME federal funds futures tick data around FOMC times

**Key Fields Needed:**
- `date`: FOMC announcement date
- `ff4_surprise`: Change in fed funds futures (scaled to basis points)
- `path_surprise`: (optional) Forward guidance component

### 1.2 Fiscal Policy Shocks (SECONDARY)

| Item | Specification |
|------|---------------|
| **Variable** | Narrative-identified government spending shocks |
| **Source** | Ramey & Zubairy (2018) |
| **Original Paper** | "Government Spending Multipliers in Good Times and in Bad," *JPE* |
| **Frequency** | Quarterly, can be used at monthly |
| **Sample** | 1889–2015 (needs updating to 2024) |
| **Construction** | News about military spending identified from Business Week, WSJ |

**Where to Get It:**
- Valerie Ramey's website: https://econweb.ucsd.edu/~vramey/research.html
- NBER working paper replication files

**Key Fields Needed:**
- `date`: Quarter
- `news_shock`: Defense news variable (% of GDP)

---

## 2. ARBITRAGE CONSTRAINT MEASURES

### 2.1 Intermediary Capital Ratio

| Item | Specification |
|------|---------------|
| **Variable** | Primary dealer equity capital / total assets |
| **Source** | He, Kelly & Manela (2017) |
| **Original Paper** | "Intermediary Asset Pricing: New Evidence from Many Asset Classes," *JFE* |
| **Frequency** | Quarterly (interpolate to monthly) |
| **Sample** | 1970–2024 |

**Where to Get It:**
- Asaf Manela's website: https://apps.olin.wustl.edu/faculty/manela/data.html
- Direct link: Look for "Intermediary Capital Risk Factor"

**Key Fields Needed:**
- `date`: Quarter-end date
- `capital_ratio`: Equity / Assets for primary dealers
- `capital_factor`: (optional) Traded factor mimicking portfolio

### 2.2 TED Spread

| Item | Specification |
|------|---------------|
| **Variable** | 3-month LIBOR minus 3-month T-bill rate |
| **Source** | Federal Reserve (FRED) |
| **Frequency** | Daily (use end-of-month) |
| **Sample** | 1986–2024 |

**Where to Get It:**
- FRED: https://fred.stlouisfed.org/series/TEDRATE
- Bloomberg: `TEDSP Index`

**Key Fields Needed:**
- `date`: Daily or monthly
- `ted_spread`: Spread in percentage points

**Note:** Post-LIBOR transition (2023+), consider using SOFR-based spread or splicing series.

### 2.3 VIX Index

| Item | Specification |
|------|---------------|
| **Variable** | CBOE Volatility Index |
| **Source** | CBOE / FRED |
| **Frequency** | Daily (use end-of-month) |
| **Sample** | 1990–2024 |

**Where to Get It:**
- FRED: https://fred.stlouisfed.org/series/VIXCLS
- CBOE: https://www.cboe.com/tradable_products/vix/
- Bloomberg: `VIX Index`

**Key Fields Needed:**
- `date`: Daily or monthly
- `vix`: Index level

---

## 3. PORTFOLIO RETURNS

### 3.1 Fama-French 5 Factors (Risk Factors)

| Item | Specification |
|------|---------------|
| **Variables** | MKT-RF, SMB, HML, RMW, CMA |
| **Source** | Kenneth French Data Library |
| **Frequency** | Monthly |
| **Sample** | 1963–2024 |

**Where to Get It:**
- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- File: "Fama/French 5 Factors (2x3)"

**Key Fields Needed:**
- `date`: Month
- `mktrf`: Market excess return
- `smb`: Size factor
- `hml`: Value factor
- `rmw`: Profitability factor
- `cma`: Investment factor
- `rf`: Risk-free rate

### 3.2 Momentum Factor

| Item | Specification |
|------|---------------|
| **Variable** | UMD (Up Minus Down) |
| **Source** | Kenneth French Data Library |
| **Frequency** | Monthly |
| **Sample** | 1963–2024 |

**Where to Get It:**
- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- File: "Momentum Factor (Mom)"

### 3.3 Additional Mispricing Portfolios

| Variable | Source | Link |
|----------|--------|------|
| Short-term reversal | Kenneth French | "Short-Term Reversal Factor" |
| Long-term reversal | Kenneth French | "Long-Term Reversal Factor" |
| Betting-Against-Beta | AQR | https://www.aqr.com/Insights/Datasets |
| Mispricing Factor (MGMT + PERF) | Robert Stambaugh | https://finance.wharton.upenn.edu/~stambaug/ |

### 3.4 Test Assets: 30 Industry Portfolios

| Item | Specification |
|------|---------------|
| **Variable** | Value-weighted returns for 30 industries |
| **Source** | Kenneth French Data Library |
| **Frequency** | Monthly |
| **Sample** | 1963–2024 |

**Where to Get It:**
- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- File: "30 Industry Portfolios"

**Industries Included:**
Food, Beer, Smoke, Games, Books, Hshld, Clths, Hlth, Chems, Txtls, Cnstr, Steel, FabPr, ElcEq, Autos, Carry, Mines, Coal, Oil, Util, Telcm, Servs, BusEq, Paper, Trans, Whlsl, Rtail, Meals, Fin, Other

---

## 4. DATA CONSTRUCTION NOTES

### 4.1 Sample Alignment

```
Binding constraint: Monetary policy shocks start 1990
Effective sample: 1990:01 – 2024:12
Observations: ~420 months
```

### 4.2 Shock Indicator Construction

```python
# Contractionary shock indicator (Equation 12 in paper)
shock_indicator = (monetary_shock < np.percentile(monetary_shock, 10)).astype(int)
```

### 4.3 Composite Constraint Indicator

```python
# Equation 12: Binary composite
c_t = (1/3) * (
    (capital_ratio < np.percentile(capital_ratio, 25)).astype(int) +  # Low capital = constrained
    (ted_spread > np.percentile(ted_spread, 75)).astype(int) +
    (vix > np.percentile(vix, 75)).astype(int)
)
```

**Note:** Capital ratio is inverted (low = constrained), TED and VIX are not (high = constrained).

### 4.4 Effective Dimension Estimation

```python
# 60-month rolling window
window = 60

# For each month t:
# 1. Get returns for months [t-59, t]
# 2. Compute covariance matrix Σ
# 3. Get eigenvalues λ_1, ..., λ_N
# 4. Compute effective dimension:

d_eff = (sum(eigenvalues)**2) / sum(eigenvalues**2)
```

---

## 5. ALTERNATIVE / ROBUSTNESS DATA

### 5.1 For Robustness Checks

| Purpose | Data | Source |
|---------|------|--------|
| Alternative shock ID | Romer & Romer (2004) shocks | AER replication files |
| Longer sample | Romer-Romer extended | Authors' website |
| Broker-dealer leverage | Adrian, Etula & Muir (2014) | NY Fed |
| Noise trader risk | Baker-Wurgler sentiment | Jeffrey Wurgler's website |
| Funding liquidity | Fontaine-Garcia bond liquidity | Bank of Canada |

### 5.2 For Extensions

| Purpose | Data | Source |
|---------|------|--------|
| International | Global factors | AQR, Kenneth French |
| Fixed income | Corporate bond indices | BAML/ICE, Bloomberg |
| Intraday | High-frequency returns | TAQ, LOBSTER |

---

## 6. QUICK START CHECKLIST

```
□ Download FF5 factors from French library
□ Download 30 industry portfolios from French library  
□ Download momentum factor from French library
□ Download UMD, ST-Rev, LT-Rev from French library
□ Download BAB from AQR
□ Download mispricing factors from Stambaugh website
□ Download VIX from FRED (VIXCLS)
□ Download TED spread from FRED (TEDRATE)
□ Download intermediary capital from Manela website
□ Download/construct Nakamura-Steinsson monetary shocks
□ Download Ramey fiscal shocks
□ Align all series to monthly frequency
□ Construct shock indicators (10th percentile threshold)
□ Construct constraint indicators (75th percentile threshold)
□ Verify sample: 1990:01 – 2024:12
```

---

## 7. CONTACT FOR DATA

| Dataset | Primary Contact |
|---------|-----------------|
| Monetary shocks | Jon Steinsson (Columbia) |
| Fiscal shocks | Valerie Ramey (UCSD) |
| Intermediary capital | Asaf Manela (WashU) |
| Mispricing factors | Robert Stambaugh (Wharton) |
| BAB factor | AQR Capital |

---

## 8. ESTIMATED EFFORT

| Task | Time |
|------|------|
| Download all public data | 2-3 hours |
| Construct/obtain shock series | 4-8 hours |
| Clean and align | 4-6 hours |
| Verify against paper | 2-4 hours |
| **Total** | **12-20 hours** |

The shock series are the bottleneck—everything else is publicly available with direct download links.
