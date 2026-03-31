import eurostat
import pandas as pd
import requests
import numpy as np
import os

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

YEARS = list(range(2014, 2023))  # 2014-2022 inclusive
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EU27 = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK"
]

ECE = ["BG", "CZ", "EE", "HR", "HU", "LT", "LV", "PL", "RO", "SI", "SK"]
WEST = [c for c in EU27 if c not in ECE]

# Actual column name returned by the eurostat package (contains backslash)
GEO_COL = "geo\\TIME_PERIOD"


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def eurostat_to_panel(dataset_code, new_col_name, filters=None):
    """
    Downloads a Eurostat dataset and returns a clean country x year panel.

    Parameters:
        dataset_code  : Eurostat dataset code (e.g. 'ilc_li02')
        new_col_name  : name for the value column in output
        filters       : dict of {dimension: value} to filter rows

    Returns:
        DataFrame with columns [country, year, new_col_name]
    """
    raw = eurostat.get_data_df(dataset_code, flags=False)

    # Apply dimension filters
    if filters:
        for dim, val in filters.items():
            if dim in raw.columns and val:
                raw = raw[raw[dim] == val]

    # Identify year columns (4-digit strings)
    year_cols = [c for c in raw.columns if str(c).isdigit() and len(str(c)) == 4]

    # Pivot wide -> long
    long = raw[[GEO_COL] + year_cols].melt(
        id_vars=[GEO_COL],
        value_vars=year_cols,
        var_name="year",
        value_name=new_col_name
    )

    # Rename geo column
    long = long.rename(columns={GEO_COL: "country"})

    # Filter to EU27 and years of interest
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long = long[long["country"].isin(EU27) & long["year"].isin(YEARS)]
    long = long.dropna(subset=[new_col_name])

    # Deduplicate by taking mean (in case filters were imperfect)
    long = long.groupby(["country", "year"])[new_col_name].mean().reset_index()

    return long


def report_missing(df):
    """Prints a missing value report for each variable column."""
    print("\n--- Missing values per variable ---")
    data_cols = [c for c in df.columns if c not in ["country", "year", "cluster"]]
    for col in data_cols:
        pct_present = (1 - df[col].isna().mean()) * 100
        status = "OK  " if pct_present >= 90 else ("WARN" if pct_present >= 70 else "MISS")
        print(f"  [{status}] {col}: {pct_present:.1f}% complete")

    print("\n--- Countries with fully missing variables ---")
    for country in sorted(EU27):
        sub = df[df["country"] == country]
        missing = [c for c in data_cols if sub[c].isna().all()]
        if missing:
            print(f"  {country}: {missing}")


# =============================================================================
# 3. BLOCK A — EUROSTAT DATA
# =============================================================================

print("\n[A] Downloading Eurostat data...")

frames = []  # collects all variable DataFrames for final merge

# --------------------------------------------------------------------------
# A1. DEPENDENT VARIABLE — Crime
# Dataset: crim_off_cat
# ICCS0101 = intentional homicide (proxy for violent crime)
# ICCS0501 + ICCS0502 = theft + burglary (property crime)
# Unit: P_HTHAB = per 100,000 inhabitants
# --------------------------------------------------------------------------

print("  -> A1: Crime (crim_off_cat)...")

try:
    crime_raw = eurostat.get_data_df("crim_off_cat", flags=False)
    year_cols = [c for c in crime_raw.columns if str(c).isdigit() and len(str(c)) == 4]

    crime_long = crime_raw[[GEO_COL, "iccs", "unit"] + year_cols].melt(
        id_vars=[GEO_COL, "iccs", "unit"],
        value_vars=year_cols,
        var_name="year",
        value_name="value"
    )
    crime_long = crime_long.rename(columns={GEO_COL: "country"})
    crime_long["year"] = pd.to_numeric(crime_long["year"], errors="coerce").astype("Int64")
    crime_long = crime_long[
        crime_long["country"].isin(EU27) &
        crime_long["year"].isin(YEARS) &
        (crime_long["unit"] == "P_HTHAB")
    ]

    # Intentional homicide
    homicide = (
        crime_long[crime_long["iccs"] == "ICCS0101"]
        .groupby(["country", "year"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "homicide_rate"})
    )
    frames.append(homicide)
    print(f"     homicide_rate: {homicide.shape[0]} obs, {homicide['country'].nunique()} countries")

    # Property crime (theft + burglary aggregated)
    property_crime = (
        crime_long[crime_long["iccs"].isin(["ICCS0501", "ICCS0502"])]
        .groupby(["country", "year"])["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "property_crime_rate"})
    )
    frames.append(property_crime)
    print(f"     property_crime_rate: {property_crime.shape[0]} obs, {property_crime['country'].nunique()} countries")

except Exception as e:
    print(f"     ERROR: {e}")


# --------------------------------------------------------------------------
# A2. MARXIAN BLOCK — Material deprivation variables
# --------------------------------------------------------------------------

print("  -> A2: Marxian block...")

marxian = [
    ("ilc_li02",    "arpr",                 {"indic_il": "LI_R_MD60", "hhtyp": "TOTAL", "unit": "PC"}),
    ("ilc_di12",    "gini",                 {"indic_il": "GINI"}),
    ("une_ltu_a",   "long_term_unemp",      {"sex": "T", "age": "Y15-74", "unit": "PC_ACT"}),
    ("ilc_mddd11",  "material_deprivation", {"hhtyp": "TOTAL"}),
    ("ilc_peps01n", "arope",                {"unit": "PC", "sex": "T", "age": "TOTAL"}),
]

for code, col, flt in marxian:
    try:
        df = eurostat_to_panel(code, col, filters=flt)
        frames.append(df)
        print(f"     {col}: {df.shape[0]} obs")
    except Exception as e:
        print(f"     ERROR {code}: {e}")


# --------------------------------------------------------------------------
# A3. WEBERIAN BLOCK — Institutional variables (Eurostat)
# --------------------------------------------------------------------------

print("  -> A3: Weberian block (Eurostat)...")

weberian_eurostat = [
    ("gov_10a_exp",  "social_expenditure", {"cofog99": "GF10", "unit": "PC_GDP", "sector": "S13", "na_item": "TE"}),
    ("demo_ndivind", "divorce_count",      {"indic_de": "DIV"}),
]

for code, col, flt in weberian_eurostat:
    try:
        df = eurostat_to_panel(code, col, filters=flt)
        frames.append(df)
        print(f"     {col}: {df.shape[0]} obs")
    except Exception as e:
        print(f"     ERROR {code}: {e}")

# Note: trust_government and trust_judiciary (sdg_16_60, sdg_16_61) are not
# available via the eurostat package. These will be added manually from
# Eurobarometer data in a later step. WGI indicators (rule_of_law,
# gov_effectiveness, control_corruption) cover the Weberian block for now.


# =============================================================================
# 4. BLOCK B — WORLD BANK WGI (direct REST API, no extra dependencies)
# =============================================================================

print("\n[B] Downloading World Bank WGI...")

iso2_to_iso3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE",
    "DE": "DEU", "DK": "DNK", "EE": "EST", "EL": "GRC", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN", "IE": "IRL",
    "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT",
    "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE",
    "SI": "SVN", "SK": "SVK"
}
iso3_to_iso2 = {v: k for k, v in iso2_to_iso3.items()}

wgi_indicators = {
    "RL.EST": "rule_of_law",
    "GE.EST": "gov_effectiveness",
    "CC.EST": "control_corruption",
}

countries_iso3 = ";".join(iso2_to_iso3.values())
year_range = f"{YEARS[0]}:{YEARS[-1]}"

for indicator, col_name in wgi_indicators.items():
    try:
        url = (
            f"https://api.worldbank.org/v2/country/{countries_iso3}"
            f"/indicator/{indicator}"
            f"?date={year_range}&format=json&per_page=1000"
        )
        response = requests.get(url, timeout=30)
        data = response.json()

        records = []
        for entry in data[1]:
            iso3 = entry.get("countryiso3code", "")
            iso2 = iso3_to_iso2.get(iso3)
            year = entry.get("date")
            value = entry.get("value")
            if iso2 and year and value is not None:
                records.append({
                    "country": iso2,
                    "year": int(year),
                    col_name: float(value)
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df["year"] = df["year"].astype("Int64")
            df = df[df["year"].isin(YEARS)]
            frames.append(df)
            print(f"     {col_name}: {df.shape[0]} obs")
        else:
            print(f"     {col_name}: no data returned")

    except Exception as e:
        print(f"     ERROR {indicator}: {e}")


# =============================================================================
# 5. BLOCK C — CPI via World Bank API (IQ.CPI.PCTY)
# Same underlying data as Transparency International, no manual download needed
# Scale: 0-100, higher = less corruption
# We invert it: corruption_index = 100 - CPI (higher = more corruption)
# =============================================================================

print("\n[C] CPI via World Bank API...")

try:
    cpi_url = (
        f"https://api.worldbank.org/v2/country/{countries_iso3}"
        f"/indicator/CC.PER.RNK"
        f"?date={year_range}&format=json&per_page=1000"
    )
    cpi_resp = requests.get(cpi_url, timeout=30).json()

    cpi_records = []
    for entry in cpi_resp[1]:
        iso2 = iso3_to_iso2.get(entry.get("countryiso3code", ""))
        year = entry.get("date")
        value = entry.get("value")
        if iso2 and year and value is not None:
            cpi_records.append({
                "country": iso2,
                "year": int(year),
                "corruption_index": 100 - float(value)
            })

    cpi_df = pd.DataFrame(cpi_records)
    cpi_df["year"] = cpi_df["year"].astype("Int64")
    cpi_df = cpi_df[cpi_df["year"].isin(YEARS)]
    frames.append(cpi_df)
    print(f"     corruption_index: {cpi_df.shape[0]} obs")

except Exception as e:
    print(f"     ERROR fetching CPI: {e}")


# =============================================================================
# 6. FINAL MERGE
# =============================================================================

print("\n[MERGE] Building final panel...")

# Full skeleton: all country x year combinations
skeleton = pd.MultiIndex.from_product(
    [EU27, YEARS], names=["country", "year"]
).to_frame(index=False)
skeleton["year"] = skeleton["year"].astype("Int64")

panel = skeleton.copy()

for df in frames:
    if df is None or df.empty:
        continue
    df = df.copy()
    df["year"] = df["year"].astype("Int64")
    # Only merge columns not already in panel (avoid duplicates)
    merge_cols = ["country", "year"] + [
        c for c in df.columns if c not in panel.columns
    ]
    if len(merge_cols) > 2:
        panel = panel.merge(df[merge_cols], on=["country", "year"], how="left")

# Normalize divorce counts to per 100,000 inhabitants using World Bank population data
if "divorce_count" in panel.columns:
    try:
        pop_url = (
            f"https://api.worldbank.org/v2/country/{countries_iso3}"
            f"/indicator/SP.POP.TOTL"
            f"?date={year_range}&format=json&per_page=1000"
        )
        pop_resp = requests.get(pop_url, timeout=30).json()
        pop_records = []
        for entry in pop_resp[1]:
            iso2 = iso3_to_iso2.get(entry.get("countryiso3code", ""))
            if iso2 and entry.get("value"):
                pop_records.append({
                    "country": iso2,
                    "year": int(entry["date"]),
                    "population": float(entry["value"])
                })
        pop_df = pd.DataFrame(pop_records)
        pop_df["year"] = pop_df["year"].astype("Int64")
        panel = panel.merge(pop_df, on=["country", "year"], how="left")
        panel["divorce_rate"] = (panel["divorce_count"] / panel["population"]) * 100000
        panel = panel.drop(columns=["divorce_count", "population"])
        print("     divorce_rate normalized to per 100k inhabitants")
    except Exception as e:
        print(f"     WARNING: could not normalize divorce_rate: {e}")
        panel = panel.rename(columns={"divorce_count": "divorce_rate"})

# Add cluster variable (ECE vs West)
panel["cluster"] = panel["country"].apply(lambda x: "ECE" if x in ECE else "WEST")

# Sort
panel = panel.sort_values(["country", "year"]).reset_index(drop=True)

print(f"\n  Panel shape   : {panel.shape[0]} rows x {panel.shape[1]} columns")
print(f"  Countries     : {panel['country'].nunique()}")
print(f"  Years         : {sorted([int(y) for y in panel['year'].unique()])}")
print(f"  Variables     : {[c for c in panel.columns if c not in ['country', 'year', 'cluster']]}")


# =============================================================================
# 7. DATA QUALITY REPORT
# =============================================================================

print("\n" + "=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)
report_missing(panel)


# =============================================================================
# 8. EXPORT
# =============================================================================

csv_path  = os.path.join(OUTPUT_DIR, "panel_eu27_raw.csv")
xlsx_path = os.path.join(OUTPUT_DIR, "panel_eu27_raw.xlsx")

panel.to_csv(csv_path, index=False)
panel.to_excel(xlsx_path, index=False, engine="openpyxl")

print(f"\n[DONE] Panel saved to:")
print(f"   {csv_path}")
print(f"   {xlsx_path}")
print("\nNext step: run 02_clean_impute.py")
