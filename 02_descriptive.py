import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

panel = pd.read_csv('data/panel_raw.csv')
panel = panel[panel['year'].between(2014, 2022)].copy()
panel.to_csv('data/panel_clean.csv', index=False)


print("=== Statistici descriptive ===")
print(panel.describe().round(2))


east = ['BG','CZ','EE','HR','HU','LV','LT','PL','RO','SI','SK']
panel['region'] = panel['country'].apply(lambda x: 'Est' if x in east else 'Vest')

print("\n=== Medii Est vs Vest ===")
print(panel.groupby('region')[['poverty_rate','gini','unemployment','crime_rate']].mean().round(2))


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

panel.groupby('year')['poverty_rate'].mean().plot(ax=axes[0], marker='o', color='steelblue')
axes[0].set_title('Rata medie a sărăciei — UE27')
axes[0].set_ylabel('%')
axes[0].set_xlabel('An')

panel.groupby('year')['crime_rate'].mean().plot(ax=axes[1], marker='o', color='firebrick')
axes[1].set_title('Rata medie a criminalității — UE27')
axes[1].set_ylabel('Infracțiuni per 100k loc.')
axes[1].set_xlabel('An')

plt.tight_layout()
plt.savefig('data/descriptive_trends.png', dpi=150)
plt.show()


plt.figure(figsize=(8, 6))
for region, grp in panel.groupby('region'):
    plt.scatter(grp['poverty_rate'], grp['crime_rate'],
                label=region, alpha=0.6, s=40)
plt.xlabel('Rata sărăciei (%)')
plt.ylabel('Rata criminalității (per 100k)')
plt.title('Sărăcie vs Criminalitate — UE27 (2014–2022)')
plt.legend()
plt.tight_layout()
plt.savefig('data/scatter_poverty_crime.png', dpi=150)
plt.show()

print("\nGrafice salvate în data/")
