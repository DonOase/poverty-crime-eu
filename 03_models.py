import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from itertools import combinations

panel = pd.read_csv('data/panel_clean.csv')

east = ['BG','CZ','EE','HR','HU','LV','LT','PL','RO','SI','SK']
panel['region'] = panel['country'].apply(lambda x: 'Est' if x in east else 'Vest')

# ── 1. Identifici outlierul ──
print("=== Top 5 țări după criminalitate medie ===")
print(panel.groupby('country')['crime_rate'].mean().sort_values(ascending=False).head(5).round(2))

# ── 2. Unit root test per țară (ADF) ──
print("\n=== ADF Test — stationaritate ===")
results = []
for country in panel['country'].unique():
    subset = panel[panel['country'] == country].sort_values('year')
    for var in ['poverty_rate', 'crime_rate']:
        series = subset[var].dropna()
        if len(series) >= 5:
            adf_stat, p_val, _, _, _, _ = adfuller(series)
            results.append({
                'country': country,
                'variable': var,
                'adf_stat': round(adf_stat, 3),
                'p_value': round(p_val, 3),
                'stationary': 'DA' if p_val < 0.05 else 'NU'
            })

adf_df = pd.DataFrame(results)
print(adf_df.groupby(['variable', 'stationary']).size().reset_index(name='count'))

# ── 3. Panel VAR + Granger causality ──
print("\n=== Granger Causality — UE27 ===")

granger_results = []

for country in panel['country'].unique():
    subset = panel[panel['country'] == country].sort_values('year')
    data = subset[['poverty_rate', 'crime_rate']].dropna()
    
    if len(data) < 6:
        continue
    
    try:
        model = VAR(data)
        fitted = model.fit(maxlags=2, ic='aic')
        
        # Granger: sărăcia cauzează criminalitatea?
        test1 = fitted.test_causality('crime_rate', ['poverty_rate'], kind='f')
        # Granger: criminalitatea cauzează sărăcia?
        test2 = fitted.test_causality('poverty_rate', ['crime_rate'], kind='f')
        
        granger_results.append({
            'country': country,
            'region': 'Est' if country in east else 'Vest',
            'poverty_causes_crime_p': round(test1.pvalue, 3),
            'crime_causes_poverty_p': round(test2.pvalue, 3),
        })
    except Exception as e:
        print(f"  {country}: eroare — {e}")

gr_df = pd.DataFrame(granger_results)
gr_df['H1_sustinuta'] = gr_df['poverty_causes_crime_p'] < 0.05
gr_df['H2_sustinuta'] = gr_df['crime_causes_poverty_p'] < 0.05

print(gr_df.to_string(index=False))

print("\n=== Sumar Granger ===")
print(f"H1 (sărăcia → criminalitatea): {gr_df['H1_sustinuta'].sum()} țări din {len(gr_df)}")
print(f"H2 (criminalitatea → sărăcia): {gr_df['H2_sustinuta'].sum()} țări din {len(gr_df)}")
print(f"\nPer regiune:")
print(gr_df.groupby('region')[['H1_sustinuta','H2_sustinuta']].sum())

gr_df.to_csv('data/granger_results.csv', index=False)
print("\nSalvat în data/granger_results.csv")
