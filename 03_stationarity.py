import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

INPUT_PATH = "data/panel_eu27_clean.csv"
OUTPUT_DIR = "data"
PLOTS_DIR  = "data/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

EU27 = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK"
]

MARXIAN  = ["arpr", "gini", "long_term_unemp", "material_deprivation", "arope"]
WEBERIAN = ["social_expenditure", "rule_of_law", "gov_effectiveness",
            "control_corruption", "corruption_index", "divorce_rate"]
OUTCOMES = ["homicide_rate", "property_crime_rate"]
ALL_VARS = OUTCOMES + MARXIAN + WEBERIAN

SIGNIFICANCE = 0.05

print("=" * 60)
print("SCRIPT 03 — STATIONARITY TESTS")
print("=" * 60)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

panel = pd.read_csv(INPUT_PATH)
panel["year"] = panel["year"].astype(int)
panel = panel.sort_values(["country", "year"]).reset_index(drop=True)

print(f"\n[LOAD] Clean panel: {panel.shape[0]} rows x {panel.shape[1]} columns")


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def adf_per_country(series, country, var):
    """
    Runs ADF test on a single country time series.
    Returns (stat, pvalue, nobs, is_stationary)
    """
    s = series.dropna()
    if len(s) < 5:
        return None, None, len(s), None
    try:
        stat, pval, _, _, _, _ = adfuller(s, autolag="AIC", regression="ct")
        return stat, pval, len(s), pval < SIGNIFICANCE
    except Exception:
        return None, None, len(s), None


def ips_test(panel_df, var):
    """
    Im-Pesaran-Shin panel unit root test (manual implementation).

    Logic: run ADF on each country, collect t-statistics, compute
    the IPS W-bar statistic and approximate p-value using the
    standard normal approximation (Choi 2001 / Im et al. 2003).

    Returns dict with test results.
    """
    t_stats = []
    country_results = []

    for country in EU27:
        series = panel_df[panel_df["country"] == country][var].values
        stat, pval, nobs, is_stat = adf_per_country(
            pd.Series(series), country, var
        )
        if stat is not None:
            t_stats.append(stat)
            country_results.append({
                "country": country,
                "adf_stat": round(stat, 4),
                "adf_pval": round(pval, 4),
                "stationary": is_stat
            })

    if not t_stats:
        return {"variable": var, "ips_stat": None, "ips_pval": None,
                "conclusion": "INSUFFICIENT DATA", "country_results": []}

    n = len(t_stats)
    t_bar = np.mean(t_stats)

    # IPS standardization constants for T=9 (approximate, from IPS 2003 Table 3)
    # E[t] and Var[t] for ADF with trend, T=9
    E_t  = -2.86   # approximate mean of ADF distribution
    Var_t = 1.73   # approximate variance

    W_bar = np.sqrt(n) * (t_bar - E_t) / np.sqrt(Var_t)

    # One-sided p-value from standard normal (H1: some panels are stationary)
    from scipy.stats import norm
    pval = norm.cdf(W_bar)  # left tail

    n_stationary = sum(1 for r in country_results if r["stationary"])

    return {
        "variable": var,
        "n_countries": n,
        "t_bar": round(t_bar, 4),
        "ips_stat": round(W_bar, 4),
        "ips_pval": round(pval, 4),
        "stationary_countries": n_stationary,
        "conclusion": "STATIONARY I(0)" if pval < SIGNIFICANCE else "NON-STATIONARY I(1)",
        "country_results": country_results
    }


def adf_on_differences(panel_df, var):
    """
    Runs IPS test on first differences of a variable.
    Used to confirm I(1) variables become stationary after differencing.
    """
    panel_diff = panel_df.copy()
    panel_diff[f"d_{var}"] = (
        panel_diff.groupby("country")[var]
        .transform(lambda x: x.diff())
    )
    return ips_test(panel_diff, f"d_{var}")


# =============================================================================
# 4. UNIT ROOT TESTS — LEVELS
# =============================================================================

print("\n[TEST] Running IPS unit root tests on levels...")
print(f"  Significance level: {SIGNIFICANCE}")
print(f"  H0: all panels contain a unit root")
print(f"  H1: some panels are stationary\n")

results_levels = []
i1_variables = []  # variables that are non-stationary in levels

for var in ALL_VARS:
    if var not in panel.columns:
        continue
    result = ips_test(panel, var)
    results_levels.append(result)

    conclusion = result["conclusion"]
    stat  = result.get("ips_stat", "N/A")
    pval  = result.get("ips_pval", "N/A")
    n_stat = result.get("stationary_countries", "N/A")

    marker = "✓" if "STATIONARY" in conclusion else "✗"
    print(f"  [{marker}] {var:<28} W={stat:<8} p={pval:<6} "
          f"({n_stat}/{result.get('n_countries','?')} countries stationary) "
          f"→ {conclusion}")

    if "NON-STATIONARY" in conclusion:
        i1_variables.append(var)


# =============================================================================
# 5. UNIT ROOT TESTS — FIRST DIFFERENCES (for I(1) variables)
# =============================================================================

if i1_variables:
    print(f"\n[TEST] Running IPS on first differences for {len(i1_variables)} I(1) variables...")

    results_diff = []
    confirmed_i1 = []
    confirmed_i2 = []

    for var in i1_variables:
        result = adf_on_differences(panel, var)
        results_diff.append(result)

        conclusion = result["conclusion"]
        stat  = result.get("ips_stat", "N/A")
        pval  = result.get("ips_pval", "N/A")

        marker = "✓" if "STATIONARY" in conclusion else "✗"
        print(f"  [{marker}] d({var:<25}) W={stat:<8} p={pval:<6} → {conclusion}")

        if "STATIONARY" in conclusion:
            confirmed_i1.append(var)
        else:
            confirmed_i2.append(var)

    if confirmed_i2:
        print(f"\n  WARNING: {confirmed_i2} appear to be I(2) — requires further inspection")
else:
    results_diff = []
    confirmed_i1 = []
    confirmed_i2 = []
    print("\n  All variables stationary in levels — no differencing needed")


# =============================================================================
# 6. INTEGRATION ORDER SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("INTEGRATION ORDER SUMMARY")
print("=" * 60)

i0_vars = [v for v in ALL_VARS if v not in i1_variables]
print(f"\n  I(0) — stationary in levels ({len(i0_vars)} variables):")
for v in i0_vars:
    print(f"    {v}")

print(f"\n  I(1) — stationary in first differences ({len(confirmed_i1)} variables):")
for v in confirmed_i1:
    print(f"    {v}")

if confirmed_i2:
    print(f"\n  I(2) — potentially ({len(confirmed_i2)} variables, INSPECT):")
    for v in confirmed_i2:
        print(f"    {v}")


# =============================================================================
# 7. DECISION: LEVELS OR DIFFERENCES FOR PVAR
# =============================================================================

print("\n" + "=" * 60)
print("PVAR MODELLING DECISION")
print("=" * 60)

all_i0 = len(i1_variables) == 0
all_i1 = len(i0_vars) == 0
mixed  = not all_i0 and not all_i1

if all_i0:
    print("""
  All variables are I(0) → use LEVELS in PVAR.
  No transformation needed.
  Decision: PVAR in levels with fixed effects.
    """)
    transform = "levels"

elif all_i1:
    print("""
  All variables are I(1) → test for cointegration (Pedroni).
  If cointegrated: VECM or PVAR in levels (Sims et al. 1990).
  If not cointegrated: PVAR in first differences.
  Decision: proceeding with first differences as baseline.
    """)
    transform = "differences"

else:
    print(f"""
  Mixed integration orders detected:
    I(0): {i0_vars}
    I(1): {confirmed_i1}

  Standard approach for mixed panels:
    Option A: Transform I(1) variables to first differences, keep I(0) in levels.
    Option B: Use PVAR in levels throughout (Sims et al. 1990 — valid for forecasting).
    Option C: Use panel ARDL / PMG which handles mixed orders natively.

  Recommended for this paper: Option A (first-difference I(1) variables).
  This is the most defensible approach for reviewers.
    """)
    transform = "mixed"


# =============================================================================
# 8. BUILD ANALYSIS-READY DATASET
# =============================================================================

print("[BUILD] Constructing analysis-ready panel...")

panel_final = panel.copy()

if transform == "differences":
    for var in ALL_VARS:
        if var in panel_final.columns:
            panel_final[f"d_{var}"] = (
                panel_final.groupby("country")[var]
                .transform(lambda x: x.diff())
            )
    # Drop first year (NaN after differencing)
    panel_final = panel_final.dropna(subset=[f"d_{var}" for var in ALL_VARS
                                             if var in panel_final.columns])
    print(f"  First differences computed. Rows after dropping NaN: {len(panel_final)}")

elif transform == "mixed":
    for var in confirmed_i1:
        if var in panel_final.columns:
            panel_final[f"d_{var}"] = (
                panel_final.groupby("country")[var]
                .transform(lambda x: x.diff())
            )
    print(f"  First differences computed for I(1) variables: {confirmed_i1}")
    print(f"  I(0) variables kept in levels: {i0_vars}")

else:
    print("  All variables in levels — no transformation applied.")

panel_final = panel_final.sort_values(["country", "year"]).reset_index(drop=True)


# =============================================================================
# 9. ACF PLOTS (visual stationarity check)
# =============================================================================

print("\n[PLOT] Generating ACF plots...")

plot_vars = OUTCOMES + MARXIAN[:2] + WEBERIAN[:2]
plot_vars = [v for v in plot_vars if v in panel.columns]

fig, axes = plt.subplots(len(plot_vars), 2, figsize=(14, len(plot_vars) * 3))

for i, var in enumerate(plot_vars):
    # Pool all country series
    series_levels = panel[var].dropna().values
    ax_lev = axes[i, 0]
    ax_lev.set_title(f"{var} — Levels", fontsize=9, fontweight="bold")
    try:
        plot_acf(series_levels, lags=6, ax=ax_lev, alpha=0.05)
    except Exception:
        ax_lev.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
    ax_lev.set_xlabel("")

    # First differences
    d_series = panel.groupby("country")[var].transform(lambda x: x.diff()).dropna().values
    ax_diff = axes[i, 1]
    ax_diff.set_title(f"Δ{var} — First Differences", fontsize=9, fontweight="bold")
    try:
        plot_acf(d_series, lags=6, ax=ax_diff, alpha=0.05)
    except Exception:
        ax_diff.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
    ax_diff.set_xlabel("")

plt.suptitle("ACF Plots — Levels vs First Differences", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/acf_plots.png", dpi=150)
plt.close()
print(f"     Saved: {PLOTS_DIR}/acf_plots.png")


# =============================================================================
# 10. EXPORT RESULTS
# =============================================================================

# Stationarity results table
rows = []
for r in results_levels:
    rows.append({
        "variable":    r["variable"],
        "test":        "IPS (levels)",
        "W_stat":      r.get("ips_stat"),
        "p_value":     r.get("ips_pval"),
        "conclusion":  r.get("conclusion"),
        "n_stationary_countries": r.get("stationary_countries")
    })
for r in results_diff:
    rows.append({
        "variable":    r["variable"],
        "test":        "IPS (first differences)",
        "W_stat":      r.get("ips_stat"),
        "p_value":     r.get("ips_pval"),
        "conclusion":  r.get("conclusion"),
        "n_stationary_countries": r.get("stationary_countries")
    })

results_df = pd.DataFrame(rows)
results_df.to_excel(f"{OUTPUT_DIR}/stationarity_results.xlsx",
                    index=False, engine="openpyxl")

# Final panel
final_path = f"{OUTPUT_DIR}/panel_eu27_final.csv"
panel_final.to_csv(final_path, index=False)

print(f"\n[DONE] Outputs saved to:")
print(f"   {OUTPUT_DIR}/stationarity_results.xlsx")
print(f"   {final_path}")
print(f"   {PLOTS_DIR}/acf_plots.png")
print("\nNext step: run 04_pvar.py")
