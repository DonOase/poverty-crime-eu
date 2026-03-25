import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Setează stilul graficelor
sns.set_theme(style="whitegrid")

def genereaza_grafice():
    print("📊 Generăm vizualizările finale pentru lucrare...")
    
    try:
        df = pd.read_csv('../data/panel_final_v2.csv')
        
        # Etichetăm grupurile pentru grafic
        df['Regiune'] = df['is_east'].map({0: 'Europa de Vest/Nord', 1: 'Europa de Est'})

        # --- GRAFIC 1: Relația Sărăcie-Furt pe Regiuni ---
        plt.figure(figsize=(12, 8))
        g = sns.lmplot(
            data=df, x="poverty_rate", y="theft_rate", hue="Regiune",
            palette="Set1", markers=["o", "s"], scatter_kws={'alpha':0.5},
            height=6, aspect=1.5
        )
        
        plt.title("Impactul Sărăciei asupra Furturilor: Est vs Vest (2008-2023)", fontsize=14)
        plt.xlabel("Rata Sărăciei (AROP %)", fontsize=12)
        plt.ylabel("Rata Furturilor (la 100k loc.)", fontsize=12)
        
        # Salvare
        plt.savefig('../data/viz_regresie_regiuni.png', dpi=300, bbox_inches='tight')
        print("✅ Grafic 1 salvat: viz_regresie_regiuni.png")

        # --- GRAFIC 2: Evoluția în timp (Media pe grupuri) ---
        plt.figure(figsize=(12, 6))
        evolutie = df.groupby(['year', 'Regiune'])['theft_rate'].mean().reset_index()
        
        sns.lineplot(data=evolutie, x="year", y="theft_rate", hue="Regiune", marker="o", linewidth=2.5)
        
        plt.title("Evoluția medie a ratelor de furt (UE27)", fontsize=14)
        plt.xlabel("An", fontsize=12)
        plt.ylabel("Media Furturilor (la 100k loc.)", fontsize=12)
        plt.xticks(df['year'].unique(), rotation=45)
        
        plt.savefig('../data/viz_evolutie_timp.png', dpi=300, bbox_inches='tight')
        print("✅ Grafic 2 salvat: viz_evolutie_timp.png")

        print("\n🚀 Vizualizările sunt gata! Verifică folderul 'data'.")

    except Exception as e:
        print(f"❌ Eroare la generarea graficelor: {e}")

if __name__ == "__main__":
    genereaza_grafice()
