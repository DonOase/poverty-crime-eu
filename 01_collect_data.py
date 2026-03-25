import eurostat
import pandas as pd
import os
import time


os.makedirs('../data', exist_ok=True)


UE27 = ['AT','BE','BG','CY','CZ','DE','DK','EE','EL','ES','FI','FR','HR','HU','IE','IT','LT','LU','LV','MT','NL','PL','PT','RO','SE','SI','SK']

def descarca_cu_verificare(cod, nume_fisier):
    cale = f'../data/{nume_fisier}.csv'
    
    
    if os.path.exists(cale):
        print(f"✅ {nume_fisier}.csv există deja.")
        return

    print(f"📥 Descarc setul de date: {cod}...")
    
    max_retries = 3
    for i in range(max_retries):
        try:
            df = eurostat.get_data_df(cod)
            
            
            if df is not None and not df.empty:
               
                df.columns = [c.split('\\')[0].split(':')[0].strip() for c in df.columns]
                
                
                tari_prezente = df['geo'].unique() if 'geo' in df.columns else []
                ue_count = len([t for t in tari_prezente if t in UE27])
                
                df.to_csv(cale, index=False)
                print(f"   → Succes! Salvat: {nume_fisier}.csv")
                print(f"   → Info: {len(df)} rânduri, {ue_count}/27 țări UE detectate.")
                break
            else:
                print(f"   ⚠️ Atenție: Setul {cod} pare a fi gol.")
        except Exception as e:
            print(f"   ❌ Eroare la încercarea {i+1}/{max_retries} pentru {cod}: {e}")
            if i < max_retries - 1:
                time.sleep(5) 
            else:
                print(f"   🛑 Am abandonat descărcarea pentru {cod}.")
    
    time.sleep(2)




descarca_cu_verificare('ilc_li02', 'raw_poverty')


descarca_cu_verificare('ilc_di12', 'raw_gini')


descarca_cu_verificare('une_rt_a', 'raw_unemployment')


descarca_cu_verificare('ilc_mddd11', 'raw_deprivation')


descarca_cu_verificare('crim_off_cat', 'raw_crime')

print("\n🚀 Etapa de colectare finalizată! Verifică folderul '../data' pentru fișiere.")
