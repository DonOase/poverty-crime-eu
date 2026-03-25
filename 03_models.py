import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Încărcare date
df = pd.read_csv('../data/panel_final_v2.csv')

# 2. Creăm variabila de interacțiune (miezul analizei tale)
# Aceasta ne spune dacă impactul sărăciei se schimbă în țările din Est
df['poverty_X_east'] = df['poverty_rate'] * df['is_east']

print("=" * 60)
print("MODELUL DE REGRESIE: SĂRĂCIE vs FURT (EFECTE FIXE)")
print("=" * 60)

# Utilizăm OLS cu 'C(geo)' pentru a simula Efectele Fixe (Entity Fixed Effects)
# Asta elimină diferențele culturale/structurale neschimbătoare dintre țări
model = smf.ols('theft_rate ~ poverty_rate + unemployment + is_east + poverty_X_east + C(geo)', data=df).fit()

print(model.summary())

# 3. Extragere coeficienți relevanți pentru interpretare
coef_poverty = model.params['poverty_rate']
coef_interaction = model.params['poverty_X_east']
p_interaction = model.pvalues['poverty_X_east']

print("\n" + "=" * 60)
print("INTERPRETARE REZULTATE:")
print(f"1. Impact general sărăcie: {coef_poverty:.4f}")
print(f"2. Diferență Est vs Vest: {coef_interaction:.4f}")

if p_interaction < 0.05:
    print(f"✅ REZULTAT: Paradoxul Estului este CONFIRMAT (p={p_interaction:.4f})")
    if coef_interaction < 0:
        print("   -> În Est, creșterea sărăciei are un impact MAI MIC asupra furtului decât în Vest.")
    else:
        print("   -> În Est, impactul sărăciei este chiar mai agresiv.")
else:
    print(f"❌ REZULTAT: Paradoxul nu este semnificativ statistic (p={p_interaction:.4f}).")
    print("   Sărăcia pare să afecteze furtul în mod similar în toată Uniunea Europeană.")

# 4. Vizualizare grafică a diferenței
plt.figure(figsize=(10, 6))
sns.lmplot(x='poverty_rate', y='theft_rate', hue='is_east', data=df, 
           palette={0: 'blue', 1: 'red'}, markers=["o", "x"], height=6)
plt.title('Relația Sărăcie - Furt: Vest (Albastru) vs Est (Roșu)')
plt.xlabel('Rata Sărăciei (%)')
plt.ylabel('Rata Furturilor (la 100k locuitori)')
plt.savefig
