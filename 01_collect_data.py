import eurostat
import pandas as pd
import os

EU27 = ['BE','BG','CZ','DK','DE','EE','IE','EL','ES','FR','HR',
        'IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT',
        'RO','SI','SK','FI','SE']

YEARS = [str(y) for y in range(2005, 2023)]

def get_clean(code, value_name, filter_dict):
    print(f"Descărcând {code}...")
    df = eurostat.get_data_df(code)
    
   
    geo_col = [c for c in df.columns if 'geo' in c.lower()][0]
    df = df.rename(columns={geo_col: 'country'})
    
   
    for col, val in filter_dict.items():
        if col in df.columns:
            df = df[df[col] == val]
    
    
    df = df[df['country'].isin(EU27)]
    
    
    year_cols = [c for c in df.columns if c in YEARS]
    df_long = df.melt(id_vars=['country'], value_vars=year_cols,
                      var_name='year', value_name=value_name)
    df_long['year'] = df_long['year'].astype(int)
    df_long = df_long.dropna(subset=[value_name])
    df_long = df_long.groupby(['country','year'])[value_name].mean().reset_index()
    
    return df_long


poverty = get_clean('ilc_li02', 'poverty_rate',
                    {'age': 'TOTAL', 'sex': 'T', 'unit': 'PC'})


gini = get_clean('ilc_di12', 'gini',
                 {'age': 'TOTAL', 'sex': 'T'})


unemployment = get_clean('une_rt_a', 'unemployment',
                         {'age': 'Y15-74', 'sex': 'T', 'unit': 'PC_ACT'})


crime = get_clean('crim_off_cat', 'crime_rate',
                  {'iccs': 'ICCS0401', 'unit': 'P_HTHAB'})


panel = poverty.merge(gini,         on=['country','year'], how='outer')
panel = panel.merge(unemployment,   on=['country','year'], how='outer')
panel = panel.merge(crime,          on=['country','year'], how='outer')

panel = panel[panel['country'].isin(EU27)]
panel = panel[panel['year'].between(2005, 2022)]
panel = panel.sort_values(['country','year']).reset_index(drop=True)

print("\n=== Primele rânduri ===")
print(panel.head(20))
print(f"\nDimensiuni: {panel.shape}")
print(f"\nValori lipsă:\n{panel.isnull().sum()}")

os.makedirs('data', exist_ok=True)
panel.to_csv('data/panel_raw.csv', index=False)
print("\nSalvat în data/panel_raw.csv")
