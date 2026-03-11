import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

panel = pd.read_csv('data/panel_clean.csv')
granger = pd.read_csv('data/granger_results.csv')

east = ['BG','CZ','EE','HR','HU','LV','LT','PL','RO','SI','SK']
panel['region'] = panel['country'].apply(lambda x: 'Est' if x in east else 'Vest')


fig, axes = plt.subplots(1, 2, figsize=(14, 8))

for ax, col, title in zip(
    axes,
    ['poverty_causes_crime_p', 'crime_causes_poverty_p'],
    ['H1: Sărăcia → Criminalitatea', 'H2: Criminalitatea → Sărăcia']
):
    granger_sorted = granger.sort_values('region')
    colors = granger_sorted[col].apply(
        lambda p: '#2ecc71' if p < 0.05 else '#e74c3c' if p > 0.10 else '#f39c12'
    )
    bars = ax.barh(granger_sorted['country'], granger_sorted[col], color=colors)
    ax.axvline(x=0.05, color='black', linestyle='--', linewidth=1.2, label='p=0.05')
    ax.axvline(x=0.10, color='gray', linestyle=':', linewidth=1.0, label='p=0.10')
    ax.set_xlabel('p-value')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)

 
    for i, (_, row) in enumerate(granger_sorted.iterrows()):
        label = '●' if row['region'] == 'Est' else '○'
        ax.text(0.97, i, label, va='center', ha='right',
                fontsize=9, color='steelblue' if row['region'] == 'Est' else 'darkorange')

green_patch = mpatches.Patch(color='#2ecc71', label='Semnificativ (p<0.05)')
orange_patch = mpatches.Patch(color='#f39c12', label='Marginal (0.05<p<0.10)')
red_patch = mpatches.Patch(color='#e74c3c', label='Nesemnificativ (p>0.10)')
axes[0].legend(handles=[green_patch, orange_patch, red_patch], loc='lower right', fontsize=8)

plt.suptitle('Teste Granger Causality — UE27 (2014–2022)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('data/granger_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(8, 5))

summary = granger.groupby('region')[['H1_sustinuta', 'H2_sustinuta']].sum().reset_index()
summary.columns = ['Regiune', 'H1: Sărăcia→Criminalitatea', 'H2: Criminalitatea→Sărăcia']

x = np.arange(len(summary))
width = 0.35

bars1 = ax.bar(x - width/2, summary['H1: Sărăcia→Criminalitatea'],
               width, label='H1: Sărăcia → Criminalitatea', color='steelblue')
bars2 = ax.bar(x + width/2, summary['H2: Criminalitatea→Sărăcia'],
               width, label='H2: Criminalitatea → Sărăcia', color='firebrick')

ax.set_xticks(x)
ax.set_xticklabels(summary['Regiune'])
ax.set_ylabel('Număr țări cu p < 0.05')
ax.set_title('Suport pentru H1 și H2 — Est vs Vest', fontsize=13, fontweight='bold')
ax.legend()
ax.set_ylim(0, 8)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(int(bar.get_height())), ha='center', va='bottom', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(int(bar.get_height())), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('data/granger_summary.png', dpi=150)
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for i, (region, grp) in enumerate(panel.groupby('region')):
    means = grp.groupby('year')[['poverty_rate', 'crime_rate']].mean()

    axes[i][0].plot(means.index, means['poverty_rate'],
                    marker='o', color='steelblue' if region == 'Est' else 'darkorange')
    axes[i][0].set_title(f'{region} — Rata sărăciei', fontweight='bold')
    axes[i][0].set_ylabel('%')
    axes[i][0].set_xlabel('An')

    axes[i][1].plot(means.index, means['crime_rate'],
                    marker='o', color='steelblue' if region == 'Est' else 'darkorange')
    axes[i][1].set_title(f'{region} — Rata criminalității', fontweight='bold')
    axes[i][1].set_ylabel('per 100k loc.')
    axes[i][1].set_xlabel('An')

plt.suptitle('Tendințe Est vs Vest — UE27 (2014–2022)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('data/trends_east_west.png', dpi=150)
plt.show()

print("Toate graficele salvate în data/")
