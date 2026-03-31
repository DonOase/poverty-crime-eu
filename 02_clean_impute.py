import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

INPUT_PATH  = "data/panel_eu27_raw.csv"
OUTPUT_DIR  = "data"
PLOTS_DIR   = "data/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

EU27 = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK"
]

ECE  = ["BG", "CZ", "EE", "HR", "HU", "LT", "LV", "PL", "RO", "SI", "SK"]
WEST = [c for c in EU27 if c not in ECE]

# Variables used in analysis
MARXIAN  = ["arpr", "gini", "long_term_unemp", "material_deprivation", "arope"]
WEBERIAN = ["social_expenditure", "rule_of_law", "gov_effectiveness",
            "control_corruption", "corruption_index", "divorce_rate"]
OUTCOMES = ["homicide_rate", "property_crime_rate"]
ALL_VARS = OUTCOMES + MARXIAN + WEBERIAN

print("=" * 60)
print("SCRIPT 02 — CLEANING AND IMPUTATION")
print("=" * 60)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

panel = pd.read_csv(INPUT_PATH)
panel["year"] = panel["year"].astype(int)
panel = panel.sort_values(["country", "year"]).reset_index(drop=True)

print(f"\n[LOAD] Raw panel: {panel.shape[0]} rows x {panel.shape[1]} columns")


# =============================================================================
# 3. MISSING VALUE HEATMAP (before imputation)
# =============================================================================

print("\n[PLOT] Generating missing value heatmap...")

fig, ax = plt.subplots(figsize=(14, 8))
missing_matrix = panel.set_index(["country", "year"])[ALL_VARS].isna().astype(int)
sns.heatmap(
    missing_matrix.T,
    cmap=["#2ecc71", "#e74c3c"],
    cbar_kws={"label": "Missing (red) / Present (green)"},
    linewidths=0.3,
    ax=ax
)
ax.set_title("Missing Values Before Imputation", fontsize=14, fontweight="bold")
ax.set_xlabel("Country × Year")
ax.set_ylabel("Variable")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/missing_heatmap_before.png", dpi=150)
plt.close()
print(f"     Saved: {PLOTS_DIR}/missing_heatmap_before.png")


# =============================================================================
# 4. IMPUTATION
# =============================================================================

print("\n[IMPUTE] Starting imputation...")

panel_clean = panel.copy()

# Fix: Italy material_deprivation values are in absolute numbers (thousands of persons)
# rather than percentages for 2014-2019, and 2020-2022 are imputed duplicates.
# Solution: set all Italy material_deprivation values to NaN and re-impute from WEST cluster mean.
print("     [FIX] Nullifying Italy material_deprivation (unit inconsistency)...")
panel_clean.loc[panel_clean["country"] == "IT", "material_deprivation"] = np.nan

for var in ALL_VARS:
    if var not in panel_clean.columns:
        print(f"     SKIP {var}: column not found")
        continue

    missing_before = panel_clean[var].isna().sum()
    if missing_before == 0:
        continue

    # --- Step 1: Linear interpolation within each country's time series ---
    # This handles sporadic gaps (e.g. one year missing between two known values)
    panel_clean[var] = (
        panel_clean
        .groupby("country")[var]
        .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
    )

    missing_after_interp = panel_clean[var].isna().sum()

    # --- Step 2: Cluster-mean imputation for fully missing country series ---
    # Used when an entire country series is missing (e.g. DE, FR, CY for social_expenditure)
    # Impute from the mean of countries in the same cluster (ECE or WEST) for each year
    if missing_after_interp > 0:
        for year in panel_clean["year"].unique():
            year_mask = panel_clean["year"] == year
            still_missing = panel_clean[var].isna() & year_mask

            if not still_missing.any():
                continue

            for country in panel_clean.loc[still_missing, "country"].unique():
                cluster = "ECE" if country in ECE else "WEST"
                cluster_countries = ECE if cluster == "ECE" else WEST
                cluster_mean = panel_clean.loc[
                    year_mask & panel_clean["country"].isin(cluster_countries),
                    var
                ].mean()

                if pd.notna(cluster_mean):
                    panel_clean.loc[
                        (panel_clean["country"] == country) & year_mask,
                        var
                    ] = cluster_mean

    missing_after_cluster = panel_clean[var].isna().sum()

    print(f"     {var}: {missing_before} missing → "
          f"{missing_after_interp} after interpolation → "
          f"{missing_after_cluster} after cluster imputation")

print(f"\n     Total missing after imputation: {panel_clean[ALL_VARS].isna().sum().sum()}")


# =============================================================================
# 5. OUTLIER DETECTION (IQR method, flagging only — not removing)
# =============================================================================

print("\n[OUTLIERS] Detecting outliers (IQR method)...")

outlier_flags = pd.DataFrame(index=panel_clean.index)

for var in ALL_VARS:
    if var not in panel_clean.columns:
        continue
    Q1 = panel_clean[var].quantile(0.25)
    Q3 = panel_clean[var].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR  # using 3x IQR for panel data (less aggressive)
    upper = Q3 + 3 * IQR
    is_outlier = (panel_clean[var] < lower) | (panel_clean[var] > upper)
    outlier_flags[f"{var}_outlier"] = is_outlier
    n_outliers = is_outlier.sum()
    if n_outliers > 0:
        offenders = panel_clean.loc[is_outlier, ["country", "year", var]].to_string(index=False)
        print(f"     {var}: {n_outliers} outlier(s)\n{offenders}\n")

# Add outlier flag column (True if ANY variable is outlier for that row)
panel_clean["any_outlier"] = outlier_flags.any(axis=1)
print(f"     Total rows flagged: {panel_clean['any_outlier'].sum()}")
print("     NOTE: outliers are FLAGGED, not removed — inspect before modelling")


# =============================================================================
# 6. Z-SCORE STANDARDIZATION
# =============================================================================

print("\n[STANDARDIZE] Z-score standardization...")

panel_std = panel_clean.copy()

for var in ALL_VARS:
    if var not in panel_std.columns:
        continue
    mean = panel_std[var].mean()
    std  = panel_std[var].std()
    panel_std[f"{var}_z"] = (panel_std[var] - mean) / std
    print(f"     {var}: mean={mean:.3f}, std={std:.3f}")


# =============================================================================
# 7. TIME SERIES PLOTS (key variables)
# =============================================================================

print("\n[PLOT] Generating time series plots...")

plot_vars = ["homicide_rate", "property_crime_rate", "arpr", "gini",
             "rule_of_law", "corruption_index"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, var in enumerate(plot_vars):
    ax = axes[i]
    for country in EU27:
        sub = panel_clean[panel_clean["country"] == country]
        color = "#e74c3c" if country in ECE else "#3498db"
        alpha = 0.6
        ax.plot(sub["year"], sub[var], color=color, alpha=alpha, linewidth=1)

    # Cluster means
    for cluster, countries, color, label in [
        ("ECE",  ECE,  "#c0392b", "ECE mean"),
        ("WEST", WEST, "#2980b9", "WEST mean"),
    ]:
        cluster_mean = (
            panel_clean[panel_clean["country"].isin(countries)]
            .groupby("year")[var].mean()
        )
        ax.plot(cluster_mean.index, cluster_mean.values,
                color=color, linewidth=2.5, linestyle="--", label=label)

    ax.set_title(var.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Legend: red = ECE, blue = WEST
fig.suptitle("Key Variables Over Time — EU27 (Red=ECE, Blue=WEST)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/time_series.png", dpi=150)
plt.close()
print(f"     Saved: {PLOTS_DIR}/time_series.png")


# =============================================================================
# 8. CORRELATION MATRIX (Marxian vs Weberian blocks)
# =============================================================================

print("\n[PLOT] Generating correlation matrix...")

corr_vars = [v for v in ALL_VARS if v in panel_clean.columns]
corr_matrix = panel_clean[corr_vars].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 8}
)
ax.set_title("Correlation Matrix — All Variables", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/correlation_matrix.png", dpi=150)
plt.close()
print(f"     Saved: {PLOTS_DIR}/correlation_matrix.png")


# =============================================================================
# 9. FINAL QUALITY CHECK
# =============================================================================

print("\n" + "=" * 60)
print("FINAL QUALITY CHECK")
print("=" * 60)

print(f"\n  Panel shape   : {panel_clean.shape[0]} rows x {panel_clean.shape[1]} columns")
print(f"  Countries     : {panel_clean['country'].nunique()}")
print(f"  Years         : {sorted(panel_clean['year'].unique().tolist())}")
print(f"  Missing values: {panel_clean[ALL_VARS].isna().sum().sum()}")
print(f"  Outlier rows  : {panel_clean['any_outlier'].sum()}")

print("\n  --- Completeness per variable ---")
for var in ALL_VARS:
    if var in panel_clean.columns:
        pct = (1 - panel_clean[var].isna().mean()) * 100
        status = "OK  " if pct >= 99 else ("WARN" if pct >= 90 else "MISS")
        print(f"  [{status}] {var}: {pct:.1f}%")


# =============================================================================
# 10. EXPORT
# =============================================================================

csv_path  = os.path.join(OUTPUT_DIR, "panel_eu27_clean.csv")
xlsx_path = os.path.join(OUTPUT_DIR, "panel_eu27_clean.xlsx")

panel_clean.to_csv(csv_path, index=False)
panel_clean.to_excel(xlsx_path, index=False, engine="openpyxl")

print(f"\n[DONE] Clean panel saved to:")
print(f"   {csv_path}")
print(f"   {xlsx_path}")
print(f"   {PLOTS_DIR}/missing_heatmap_before.png")
print(f"   {PLOTS_DIR}/time_series.png")
print(f"   {PLOTS_DIR}/correlation_matrix.png")
print("\nNext step: run 03_stationarity.py")
