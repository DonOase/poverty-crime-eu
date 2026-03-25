import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

warnings.filterwarnings('ignore')

def test_causalitate():
    print("=" * 60)
    print("TESTUL DE CAUZALITATE GRANGER (Direcția Relației)")
    print("=" * 60)

    try:
        df = pd.read_csv('../data/panel_final_v2.csv')
        
        # Agregăm datele la nivel de UE pentru o serie temporală continuă
        # Testul Granger funcționează cel mai bine pe serii de timp consolidate
        ts_data = df.groupby('year')[['theft_rate', 'poverty_rate', 'unemployment']].mean()

        variables = {
            'Sărăcie -> Furt': ['theft_rate', 'poverty_rate'],
            'Șomaj -> Furt': ['theft_rate', 'unemployment']
        }

        for label, cols in variables.items():
            print(f"\n🔍 Testăm: {label}")
            # Testăm pentru un decalaj (lag) de 1 și 2 ani
            # (Infracțiunile de subzistență apar de obicei repede după șocul economic)
            results = grangercausalitytests(ts_data[cols], maxlag=2, verbose=False)
            
            # Extragem p-value pentru testul F la lag 1
            p_val_lag1 = results[1][0]['ssr_ftest'][1]
            p_val_lag2 = results[2][0]['ssr_ftest'][1]

            print(f"   P-value (Lag 1 an):  {p_val_lag1:.4f}")
            print(f"   P-value (Lag 2 ani): {p_val_lag2:.4f}")

            if p_val_lag1 < 0.05 or p_val_lag2 < 0.05:
                print(f"   ✅ REZULTAT: Există o relație de cauzalitate predictivă.")
            else:
                print(f"   ❌ REZULTAT: Nu există cauzalitate Granger (variabilele sunt independente).")

    except Exception as e:
        print(f"❌ Eroare la testul Granger: {e}")

if __name__ == "__main__":
    test_causalitate()
