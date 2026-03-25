import pandas as pd
import numpy as np
import os

# Setări directoare
DATA_DIR = '../data'
OUTPUT_PANEL = os.path.join(DATA_DIR, 'panel_final_v2.csv')

# Lista oficială UE27 (fără UK)
UE27 = ['AT','BE','BG','CY','CZ','DE','DK','EE','EL','ES','FI','FR','HR','HU','IE','IT','LT','LU','LV','MT','NL','PL','PT','RO','SE','SI','SK']

def load_eurostat_csv(filename, value_name, filters=None):
    path = os.path.join(DATA_DIR, f"{filename}.csv")
    if not os.path.exists(path):
        print(f"⚠️ Fișierul {filename}.csv lipsește!")
        return pd.DataFrame()

    df = pd.read_csv(path)
    
    # Curățăm numele coloanelor (Eurostat pune uneori 'geo\\TIME_PERIOD')
    df.columns = [c.split('\\')[0].split(':')[0].strip() for c in df.columns]
    
    # Aplicăm filtrele (unitate, sex, vârstă etc.)
    if filters:
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col] == val]
    
    # Identificăm coloanele care reprezintă ani
    years = [c for c in df.columns if str(c).isdigit()]
    
    # Pivotăm tabelul de la format lat la format lung
    df_long = df.melt(id_vars=['geo'], value_vars=years, var_name='year', value_name=value_name)
    
    # Convertim anii în numere și valorile în format numeric
    df_long['year'] = df_long['year'].astype(int)
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce')
    
    # Filtrăm doar țările UE27
    return df_long[df_long['geo'].isin(UE27)]

print("🔄 Pasul 1: Încărcare și curățare date brute...")

# 1. Rata Sărăciei
poverty = load_eurostat_csv('raw_poverty', 'poverty_rate', 
                           filters={'indic_il': 'LI_R_M60', 'unit': 'PC', 'sex': 'T', 'age': 'TOTAL'})

# 2. Rata Criminalității (Furturi)
crime = load_eurostat_csv('raw_crime', 'theft_rate', 
                         filters={'iccs': 'ICCS0501', 'unit': 'P_HTHAB'})

# 3. Șomajul (variabilă de control importantă)
unemployment = load_eurostat_csv('raw_unemployment', 'unemployment', 
                                filters={'age': 'Y15-74', 'unit': 'PC_ACT', 'sex': 'T'})

# --- COMBINARE (MERGE) ---
print("🔗 Pasul 2: Combinare seturi de date...")
# Combinăm Sărăcia cu Criminalitatea
panel = pd.merge(crime, poverty, on=['geo', 'year'], how='inner')
# Adăugăm și Șomajul
panel = pd.merge(panel, unemployment, on=['geo', 'year'], how='left')

# --- IMPUTARE / INTERPOLARE ---
# Sortăm pentru a asigura ordinea cronologică per țară
panel = panel.sort_values(['geo', 'year'])

# Completăm gaurile mici de date (max 2 ani) pentru a nu pierde observații
for col in ['poverty_rate', 'theft_rate', 'unemployment']:
    panel[col] = panel.groupby('geo')[col].transform(lambda x: x.interpolate(limit=2))

# --- VARIABILE FINALE ---
# Adăugăm variabila Dummy pentru Europa de Est
east_europe = ['BG', 'CZ', 'EE', 'HR', 'HU', 'LT', 'LV', 'PL', 'RO', 'SI', 'SK']
panel['is_east'] = panel['geo'].apply(lambda x: 1 if x in east_europe else 0)

# Resetăm indexul pentru a ne asigura că 'geo' și 'year' sunt COLOANE simple
panel = panel.reset_index(drop=True)

# --- SALVARE ---
panel.to_csv(OUTPUT_PANEL, index=False)

print("-" * 30)
print(f"✅ SUCCES! Fișier generat: {OUTPUT_PANEL}")
print(f"📊 Total rânduri: {len(panel)}")
print(f"🌍 Țări în panel: {panel['geo'].nunique()}")
print(f"📅 Interval ani: {panel['year'].min()} - {panel['year'].max()}")
print("-" * 30)
