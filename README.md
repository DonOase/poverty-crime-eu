# Poverty & Crime in the EU — Computational Sociology Project

## Research Question
Do EU countries steal because they are poor, or are they poor because they steal?
A Granger causality analysis testing the Marxian (structural) vs. Weberian (cultural)
paradigm on panel data for 27 EU member states, 2014–2022.

## Hypotheses
- **H1 (Marxian):** Poverty Granger-causes crime
- **H2 (Weberian):** Crime Granger-causes poverty
- **H3 (Heterogeneity):** The causal direction differs between Eastern and Western EU

## Data Sources
| Variable | Indicator | Source |
|---|---|---|
| Poverty rate | `ilc_li02` — at-risk-of-poverty rate | Eurostat |
| Gini coefficient | `ilc_di12` | Eurostat |
| Unemployment | `une_rt_a` — total, 15–74, % active pop. | Eurostat |
| Crime rate | `crim_off_cat` — theft (ICCS0401), per 100k | Eurostat |

**Coverage:** 27 EU countries × 9 years (2014–2022) = 243 observations, balanced panel, 0 missing values.

**Note on Gini:** Available only from 2014 onward on Eurostat (`ilc_di12`).
This limits the panel to 2014–2022. Acknowledged as limitation in the paper.

**Note on Romania:** Excluded from Granger causality models — VAR selected 0 lags
(insufficient temporal variation). Retained in descriptive statistics.

## Scripts — Run in Order

### `01_collect_data.py`
Downloads all variables from Eurostat API using the `eurostat` Python package.
Merges into a single panel DataFrame.
Output: `data/panel_raw.csv`

### `02_descriptive.py`
Computes descriptive statistics and produces exploratory visualizations.
Splits countries into East vs. West EU groups.
Output: `data/descriptive_trends.png`, `data/scatter_poverty_crime.png`

### `03_models.py`
Runs ADF unit root tests per country and per variable.
Estimates VAR models and Granger causality tests (F-test) for each country.
Output: `data/granger_results.csv`

### `04_visualize.py`
Produces publication-ready figures from Granger results.
Output: `data/granger_heatmap.png`, `data/granger_summary.png`, `data/trends_east_west.png`

## Key Findings
1. **Structural paradox:** Eastern EU countries have higher poverty but significantly
   lower crime rates than Western EU — contradicting a simple linear Marx hypothesis.
2. **No universal causality:** In 18 out of 26 countries, neither H1 nor H2 is
   supported (p > 0.05). Context matters more than any universal mechanism.
3. **Regional asymmetry (H3 supported):** In the East, H2 dominates (3 countries vs. 1).
   Post-communist dynamics suggest crime/corruption perpetuated poverty, not vice versa.

## Dependencies
```bash
pip install eurostat pandas numpy matplotlib seaborn statsmodels
```

## Target Journal
Social Science Research (Elsevier) — SSCI Q1, IF ~4.1
Fallback: European Sociological Review (Oxford) — SSCI Q1

## Project Status
- [x] Data collection
- [x] Descriptive statistics
- [x] Granger causality models
- [x] Visualizations
- [ ] Theoretical framework (writing)
- [ ] Results section (writing)
- [ ] Discussion & conclusions (writing)
- [ ] Submission
