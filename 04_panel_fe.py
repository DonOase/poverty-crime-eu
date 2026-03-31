import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings("ignore")

from linearmodels.panel import PanelOLS
from scipy import stats

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

INPUT_PATH  = "data/panel_eu27_clean.csv"
RESULTS_DIR = "data/results"
PLOTS_DIR   = "data/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

EU27 = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK"
]
ECE  = ["BG", "CZ", "EE", "HR", "HU", "LT", "LV", "PL", "RO", "SI", "SK"]
WEST = [c for c in EU27 if c not in ECE]

# Final variable selection (post multicollinearity check)
OUTCOMES = ["homicide_rate", "property_crime_rate"]
MARXIAN  = ["arpr", "long_term_unemp"]
WEBERIAN = ["rule_of_law", "social_expenditure", "divorce_rate"]
ALL_PREDICTORS = MARXIAN + WEBERIAN

print("=" * 60)
print("SCRIPT 04 — PANEL FIXED EFFECTS MODELS")
print("=" * 60)


# =============================================================================
# 2. LOAD AND PREPARE DATA
# =============================================================================

panel = pd.read_csv(INPUT_PATH)
panel["year"] = panel["year"].astype(int)
panel = panel.sort_values(["country", "year"]).reset_index(drop=True)

# Log-transform outcomes (standard in crime literature — reduces skewness)
# Add small constant to avoid log(0)
for outcome in OUTCOMES:
    panel[f"log_{outcome}"] = np.log(panel[outcome] + 0.01)

LOG_OUTCOMES = [f"log_{o}" for o in OUTCOMES]

print(f"\n[LOAD] Panel: {panel.shape[0]} rows x {panel.shape[1]} columns")
print(f"  Countries: {panel['country'].nunique()}")
print(f"  Years: {sorted(panel['year'].unique().tolist())}")
print(f"  Outcomes: {LOG_OUTCOMES}")
print(f"  Marxian predictors: {MARXIAN}")
print(f"  Weberian predictors: {WEBERIAN}")


# =============================================================================
# 3. SET PANEL INDEX (required by linearmodels)
# =============================================================================

def prepare_panel(df):
    """Sets MultiIndex [country, year] required by linearmodels PanelOLS."""
    df = df.copy()
    df = df.set_index(["country", "year"])
    return df


# =============================================================================
# 4. MODEL ESTIMATION FUNCTION
# =============================================================================

def estimate_fe_model(df, outcome, predictors, label=""):
    """
    Estimates a Fixed Effects model with Driscoll-Kraay standard errors.

    Parameters:
        df         : DataFrame with MultiIndex [country, year]
        outcome    : dependent variable name
        predictors : list of independent variable names
        label      : model label for printing

    Returns:
        dict with coefficients, standard errors, p-values, R2, N
    """
    # Drop rows with any missing value in outcome or predictors
    cols = [outcome] + predictors
    data = df[cols].dropna()

    if len(data) < 20:
        print(f"  [{label}] Insufficient observations ({len(data)}) — skipping")
        return None

    # Build formula
    formula = f"{outcome} ~ " + " + ".join(predictors) + " + EntityEffects"

    try:
        model  = PanelOLS.from_formula(formula, data=data)
        result = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=2)

        # Extract results
        coef_df = pd.DataFrame({
            "coefficient": result.params,
            "std_error":   result.std_errors,
            "t_stat":      result.tstats,
            "p_value":     result.pvalues,
            "ci_lower":    result.params - 1.96 * result.std_errors,
            "ci_upper":    result.params + 1.96 * result.std_errors,
        })

        # Significance stars
        def stars(p):
            if p < 0.01:  return "***"
            if p < 0.05:  return "**"
            if p < 0.10:  return "*"
            return ""

        coef_df["stars"] = coef_df["p_value"].apply(stars)

        output = {
            "label":       label,
            "outcome":     outcome,
            "predictors":  predictors,
            "n_obs":       result.nobs,
            "n_entities":  result.entity_info.total,
            "r2_within":   result.rsquared,
            "r2_between":  result.rsquared_between if hasattr(result, "rsquared_between") else None,
            "f_stat":      result.f_statistic.stat if hasattr(result.f_statistic, "stat") else None,
            "f_pval":      result.f_statistic.pval if hasattr(result.f_statistic, "pval") else None,
            "coefficients": coef_df,
            "result_obj":  result
        }

        # Print summary
        print(f"\n  [{label}] {outcome} ~ {' + '.join(predictors)}")
        print(f"  N={result.nobs}, Entities={result.entity_info.total}, R²(within)={result.rsquared:.3f}")
        print(f"  {'Variable':<25} {'Coef':>8} {'SE':>8} {'t':>7} {'p':>7} {'Sig':>4}")
        print(f"  {'-'*60}")
        for var, row in coef_df.iterrows():
            print(f"  {str(var):<25} {row['coefficient']:>8.4f} {row['std_error']:>8.4f} "
                  f"{row['t_stat']:>7.3f} {row['p_value']:>7.4f} {row['stars']:>4}")

        return output

    except Exception as e:
        print(f"  [{label}] ERROR: {e}")
        return None


# =============================================================================
# 5. FULL SAMPLE MODELS (H1 and H2)
# =============================================================================

print("\n" + "=" * 60)
print("FULL SAMPLE — EU27")
print("=" * 60)

panel_idx = prepare_panel(panel)
results_full = []

model_specs = [
    # (label, outcome, predictors)
    ("M1", "log_homicide_rate",       MARXIAN),
    ("M2", "log_homicide_rate",       WEBERIAN),
    ("M3", "log_homicide_rate",       ALL_PREDICTORS),
    ("M4", "log_property_crime_rate", MARXIAN),
    ("M5", "log_property_crime_rate", WEBERIAN),
    ("M6", "log_property_crime_rate", ALL_PREDICTORS),
]

for label, outcome, predictors in model_specs:
    res = estimate_fe_model(panel_idx, outcome, predictors, label)
    if res:
        results_full.append(res)


# =============================================================================
# 6. ECE SUBSAMPLE (H3 — Eastern Europe)
# =============================================================================

print("\n" + "=" * 60)
print("ECE SUBSAMPLE")
print("=" * 60)

panel_ece     = panel[panel["country"].isin(ECE)].copy()
panel_ece_idx = prepare_panel(panel_ece)
results_ece   = []

for label, outcome, predictors in model_specs:
    res = estimate_fe_model(panel_ece_idx, outcome, predictors, f"{label}_ECE")
    if res:
        results_ece.append(res)


# =============================================================================
# 7. WEST SUBSAMPLE (H3 — Western Europe)
# =============================================================================

print("\n" + "=" * 60)
print("WEST SUBSAMPLE")
print("=" * 60)

panel_west     = panel[panel["country"].isin(WEST)].copy()
panel_west_idx = prepare_panel(panel_west)
results_west   = []

for label, outcome, predictors in model_specs:
    res = estimate_fe_model(panel_west_idx, outcome, predictors, f"{label}_WEST")
    if res:
        results_west.append(res)


# =============================================================================
# 8. GRANGER CAUSALITY TESTS (panel, bivariate)
# =============================================================================

print("\n" + "=" * 60)
print("GRANGER CAUSALITY TESTS")
print("=" * 60)
print("  Testing: do predictors Granger-cause crime outcomes?")
print("  Method: panel bivariate OLS with lagged predictors\n")

def panel_granger_test(df, cause, effect, lags=1):
    """
    Simple panel Granger causality test.
    Tests whether lagged values of 'cause' predict 'effect'
    after controlling for lagged 'effect' (with entity fixed effects).

    H0: 'cause' does NOT Granger-cause 'effect'
    Reject H0 if p < 0.05
    """
    df = df.copy().reset_index()
    df = df.sort_values(["country", "year"])

    # Create lagged variables
    for lag in range(1, lags + 1):
        df[f"{effect}_lag{lag}"]  = df.groupby("country")[effect].shift(lag)
        df[f"{cause}_lag{lag}"]   = df.groupby("country")[cause].shift(lag)

    df = df.dropna()
    if len(df) < 20:
        return None, None, None

    # Demean by entity (within transformation = fixed effects)
    for col in [effect] + [f"{effect}_lag{l}" for l in range(1, lags+1)] + \
               [f"{cause}_lag{l}" for l in range(1, lags+1)]:
        df[f"{col}_dm"] = df[col] - df.groupby("country")[col].transform("mean")

    y_col    = f"{effect}_dm"
    x_effect = [f"{effect}_lag{l}_dm" for l in range(1, lags+1)]
    x_cause  = [f"{cause}_lag{l}_dm"  for l in range(1, lags+1)]

    # Restricted model (only lagged effect)
    X_r = df[x_effect].values
    X_r = np.column_stack([np.ones(len(X_r)), X_r])
    y   = df[y_col].values

    # Unrestricted model (lagged effect + lagged cause)
    X_u = df[x_effect + x_cause].values
    X_u = np.column_stack([np.ones(len(X_u)), X_u])

    try:
        # OLS both models
        beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]

        rss_r = np.sum((y - X_r @ beta_r) ** 2)
        rss_u = np.sum((y - X_u @ beta_u) ** 2)

        n  = len(y)
        k  = lags          # number of restrictions
        df_u = n - X_u.shape[1]

        f_stat = ((rss_r - rss_u) / k) / (rss_u / df_u)
        p_val  = 1 - stats.f.cdf(f_stat, k, df_u)

        return f_stat, p_val, n

    except Exception:
        return None, None, None


granger_results = []

# Test all predictor -> outcome combinations
for outcome in LOG_OUTCOMES:
    for predictor in ALL_PREDICTORS:
        for sample_name, sample_df in [
            ("EU27", panel),
            ("ECE",  panel_ece),
            ("WEST", panel_west)
        ]:
            f_stat, p_val, n = panel_granger_test(
                sample_df.set_index(["country", "year"]).reset_index(),
                cause=predictor,
                effect=outcome,
                lags=1
            )

            if f_stat is not None:
                sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
                granger_results.append({
                    "cause":    predictor,
                    "effect":   outcome,
                    "sample":   sample_name,
                    "f_stat":   round(f_stat, 4),
                    "p_value":  round(p_val, 4),
                    "n_obs":    n,
                    "block":    "Marxian" if predictor in MARXIAN else "Weberian",
                    "sig":      sig
                })

                marker = "→ GRANGER CAUSES" if p_val < 0.05 else "  no effect"
                print(f"  [{sample_name}] {predictor:<25} → {outcome:<28} "
                      f"F={f_stat:.3f} p={p_val:.4f} {sig} {marker}")

granger_df = pd.DataFrame(granger_results)


# =============================================================================
# 9. COEFFICIENT PLOT
# =============================================================================

print("\n[PLOT] Generating coefficient plots...")

def plot_coefficients(results_list, title, filename):
    """Plots coefficients with confidence intervals for a list of model results."""

    # Only full models (M3, M6)
    full_models = [r for r in results_list if r["label"].endswith("3") or r["label"].endswith("6")]
    if not full_models:
        full_models = results_list

    n_models = len(full_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 7), sharey=False)

    if n_models == 1:
        axes = [axes]

    colors = {
        "Marxian":  "#e74c3c",
        "Weberian": "#3498db",
    }

    for ax, res in zip(axes, full_models):
        coef_df = res["coefficients"].copy()

        # Assign colors by block
        block_colors = []
        for var in coef_df.index:
            if var in MARXIAN:
                block_colors.append(colors["Marxian"])
            elif var in WEBERIAN:
                block_colors.append(colors["Weberian"])
            else:
                block_colors.append("#95a5a6")

        y_pos = range(len(coef_df))
        ax.barh(
            y_pos,
            coef_df["coefficient"],
            xerr=[
                coef_df["coefficient"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["coefficient"]
            ],
            color=block_colors,
            alpha=0.8,
            capsize=4,
            height=0.6
        )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(coef_df.index, fontsize=9)
        ax.set_title(f"{res['label']}: {res['outcome']}\nR²={res['r2_within']:.3f}, N={res['n_obs']}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Coefficient (with 95% CI)")
        ax.grid(True, alpha=0.3, axis="x")

        # Add significance stars
        for i, (var, row) in enumerate(coef_df.iterrows()):
            if row["stars"]:
                ax.text(
                    row["ci_upper"] + 0.001,
                    i,
                    row["stars"],
                    va="center",
                    fontsize=10,
                    color="black"
                )

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors["Marxian"],  label="Marxian block"),
        mpatches.Patch(color=colors["Weberian"], label="Weberian block"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {PLOTS_DIR}/{filename}")


plot_coefficients(results_full, "Fixed Effects — Full Sample EU27", "coef_full.png")
plot_coefficients(results_ece,  "Fixed Effects — ECE Subsample",    "coef_ece.png")
plot_coefficients(results_west, "Fixed Effects — WEST Subsample",   "coef_west.png")


# =============================================================================
# 10. EXPORT RESULTS TO EXCEL
# =============================================================================

print("\n[EXPORT] Saving results...")

def results_to_df(results_list):
    """Converts list of model result dicts to a flat DataFrame for export."""
    rows = []
    for res in results_list:
        coef_df = res["coefficients"]
        for var, row in coef_df.iterrows():
            rows.append({
                "model":       res["label"],
                "outcome":     res["outcome"],
                "variable":    var,
                "coefficient": round(row["coefficient"], 6),
                "std_error":   round(row["std_error"], 6),
                "t_stat":      round(row["t_stat"], 4),
                "p_value":     round(row["p_value"], 4),
                "ci_lower":    round(row["ci_lower"], 6),
                "ci_upper":    round(row["ci_upper"], 6),
                "stars":       row["stars"],
                "n_obs":       res["n_obs"],
                "r2_within":   round(res["r2_within"], 4),
            })
    return pd.DataFrame(rows)


with pd.ExcelWriter(f"{RESULTS_DIR}/fe_results.xlsx", engine="openpyxl") as writer:
    results_to_df(results_full).to_excel(writer, sheet_name="EU27_full",  index=False)
    results_to_df(results_ece).to_excel(writer,  sheet_name="ECE",        index=False)
    results_to_df(results_west).to_excel(writer, sheet_name="WEST",       index=False)
    granger_df.to_excel(writer,                  sheet_name="Granger",    index=False)

print(f"   {RESULTS_DIR}/fe_results.xlsx")
print(f"   {PLOTS_DIR}/coef_full.png")
print(f"   {PLOTS_DIR}/coef_ece.png")
print(f"   {PLOTS_DIR}/coef_west.png")
print("\n[DONE] Next step: run 05_tables.py")
