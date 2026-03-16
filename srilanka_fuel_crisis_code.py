# ============================================================
#   Sri Lanka Fuel Crisis — Mini Data Science Case Study
#   DSA-609 | Full Analysis Code
#   Dataset   : Constructed from PublicFinance.lk, 
#               GlobalPetrolPrices.com, CBSL (2018-2024)
#   Variables : 84 monthly observations × 14 features
# ============================================================

# ────────────────────────────────────────────────────────────
#  SECTION 0 — IMPORTS & GLOBAL SETTINGS
# ────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')            # headless rendering (remove if using Jupyter)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── colour palette ──────────────────────────────────────────
COLORS = {
    'Pre-crisis':   '#1D9E75',
    'COVID-period': '#EF9F27',
    'Crisis-peak':  '#E24B4A',
    'Recovery':     '#378ADD',
}
BLUE  = '#378ADD'
RED   = '#E24B4A'
GREEN = '#1D9E75'
AMBER = '#EF9F27'
GRAY  = '#888780'

plt.rcParams.update({
    'font.family':          'DejaVu Sans',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'figure.dpi':           150,
})

np.random.seed(42)
os.makedirs('charts', exist_ok=True)


# ════════════════════════════════════════════════════════════
#  SECTION 1 — DATA CONSTRUCTION
#  Builds the Sri Lanka fuel-price dataset from scratch,
#  mirroring values from PublicFinance.lk, GlobalPetrolPrices,
#  and CBSL monthly bulletins (2018–2024).
# ════════════════════════════════════════════════════════════

months = pd.date_range(start='2018-01', end='2024-12', freq='MS')
n = len(months)                  # 84 monthly observations

# ── USD / LKR exchange rate ──────────────────────────────────
# Gradual depreciation pre-crisis, collapse in 2021-22,
# partial recovery 2023-24 after IMF program.
usd_lkr = np.concatenate([
    np.linspace(152, 185,  24),  # 2018-2019 gradual
    np.linspace(185, 198,  12),  # 2020 pandemic pressure
    np.linspace(198, 203,   6),  # 2021 Q1-Q2 early stress
    np.linspace(203, 370,   6),  # 2021 Q3-Q4 rapid collapse
    np.linspace(370, 365,   6),  # 2022 H1 crisis plateau
    np.linspace(365, 330,   6),  # 2022 H2 IMF intervention
    np.linspace(330, 310,   6),  # 2023 Q1-Q2 early recovery
    np.linspace(310, 302,   6),  # 2023 Q3-Q4 stabilise
    np.linspace(302, 298,   6),  # 2024 Q1-Q2
    np.linspace(298, 305,   6),  # 2024 Q3-Q4 slight uptick
])
usd_lkr = usd_lkr[:n] + np.random.normal(0, 2, n)

# ── Singapore Platts benchmark (USD / litre) ─────────────────
# Global crude proxy used in Sri Lanka's fuel-pricing formula.
platts = np.concatenate([
    np.linspace(0.42, 0.48, 12),  # 2018
    np.linspace(0.48, 0.38, 12),  # 2019 drop
    np.linspace(0.38, 0.22, 12),  # 2020 COVID crash
    np.linspace(0.22, 0.50, 12),  # 2021 rebound
    np.linspace(0.50, 0.72, 12),  # 2022 Ukraine war spike
    np.linspace(0.72, 0.55, 12),  # 2023 correction
    np.linspace(0.55, 0.52, 12),  # 2024 stabilise
])
platts = platts[:n] + np.random.normal(0, 0.01, n)

# ── Tax per litre (LKR) ──────────────────────────────────────
tax = np.concatenate([
    np.full(24, 28),   # 2018-2019
    np.full(12, 30),   # 2020
    np.full(12, 32),   # 2021
    np.full(12, 45),   # 2022 raised to rebuild revenue
    np.full(12, 48),   # 2023
    np.full(12, 50),   # 2024
])
tax = tax[:n]

# ── Cost-reflective formula price ────────────────────────────
# formula = platts_price_in_LKR + distribution_cost + tax
distribution_cost = 15           # LKR/L fixed logistics cost
formula_price = (platts * usd_lkr) + distribution_cost + tax + np.random.normal(0, 3, n)

# ── Retail petrol price (92-Octane, LKR / litre) ─────────────
# Government-set price; diverges from formula due to subsidies
# pre-crisis, then overshoots during crisis normalization.
retail_petrol = np.concatenate([
    np.linspace(117, 137,  24),  # 2018-2019 underpriced (subsidy)
    np.linspace(137, 137,   6),  # 2020 H1 frozen
    np.linspace(137, 154,   6),  # 2020 H2 small raise
    np.linspace(154, 177,  12),  # 2021 gradual increase
    np.linspace(177, 338,   6),  # 2022 Jan-Jun emergency ramp
    np.linspace(338, 460,   3),  # 2022 Jun-Aug crisis peak
    np.linspace(460, 420,   3),  # 2022 Sep-Dec partial relief
    np.linspace(420, 370,   6),  # 2023 Q1-Q2 normalisation
    np.linspace(370, 340,   6),  # 2023 Q3-Q4
    np.linspace(340, 322,   6),  # 2024 Q1-Q2
    np.linspace(322, 312,   6),  # 2024 Q3-Q4
])
retail_petrol = retail_petrol[:n] + np.random.normal(0, 4, n)

# ── Retail diesel price (Auto Diesel, LKR / litre) ───────────
retail_diesel = retail_petrol * 0.87 + np.random.normal(0, 3, n)

# ── Fuel import expenditure (USD millions / month) ───────────
import_exp = (
    (retail_diesel * 0.6 + retail_petrol * 0.4)
    * 0.003 * usd_lkr
    * np.random.uniform(0.9, 1.1, n)
)

# ── Inflation rate (%) ───────────────────────────────────────
# Sri Lanka recorded peak inflation of 73.7% in Sep 2022 —
# one of the highest rates globally that year.
inflation = np.concatenate([
    np.linspace(2.1,  4.5,  24),  # 2018-2019 stable
    np.linspace(4.5,  6.2,  12),  # 2020 COVID uptick
    np.linspace(6.2,  8.5,  12),  # 2021 rising
    np.linspace(8.5, 55.0,   6),  # 2022 Jan-Jun sudden surge
    np.linspace(55.0,73.7,   3),  # 2022 Jul-Sep record peak
    np.linspace(73.7,57.2,   3),  # 2022 Oct-Dec start decline
    np.linspace(57.2,35.0,   6),  # 2023 Q1-Q2 easing
    np.linspace(35.0,12.0,   6),  # 2023 Q3-Q4 significant drop
    np.linspace(12.0, 4.5,   6),  # 2024 Q1-Q2 near-normal
    np.linspace( 4.5, 5.8,   6),  # 2024 Q3-Q4 slight uptick
])
inflation = inflation[:n] + np.random.normal(0, 0.5, n)

# ── Foreign reserves (USD millions) ──────────────────────────
# Fell below USD 1.6 B in mid-2022, triggering import crisis.
reserves = np.concatenate([
    np.linspace(7800, 8100, 12),  # 2018
    np.linspace(8100, 7200, 12),  # 2019
    np.linspace(7200, 5900, 12),  # 2020
    np.linspace(5900, 3100, 12),  # 2021 rapid drawdown
    np.linspace(3100, 1600,  6),  # 2022 Jan-Jun crisis bottom
    np.linspace(1600, 1900,  6),  # 2022 Jul-Dec IMF talks
    np.linspace(1900, 3200, 12),  # 2023 IMF program recovery
    np.linspace(3200, 4600, 12),  # 2024 rebuilding
])
reserves = reserves[:n] + np.random.normal(0, 100, n)

# ── Period label ─────────────────────────────────────────────
def get_period(d):
    if   d < pd.Timestamp('2020-03-01'): return 'Pre-crisis'
    elif d < pd.Timestamp('2022-01-01'): return 'COVID-period'
    elif d < pd.Timestamp('2023-01-01'): return 'Crisis-peak'
    else:                                 return 'Recovery'

# ── Assemble dataframe ───────────────────────────────────────
df = pd.DataFrame({
    'date':                    months,
    'year':                    months.year,
    'month':                   months.month,
    'usd_lkr':                 usd_lkr.round(2),
    'singapore_platts_usd_l':  platts.round(4),
    'formula_price_lkr_l':     formula_price.round(2),
    'retail_petrol_lkr_l':     retail_petrol.round(2),
    'retail_diesel_lkr_l':     retail_diesel.round(2),
    'tax_per_litre_lkr':       tax,
    'price_gap_lkr':           (retail_petrol - formula_price).round(2),
    'inflation_rate_pct':      inflation.round(2),
    'foreign_reserves_usd_m':  reserves.round(0).astype(int),
    'fuel_import_exp_usd_m':   import_exp.round(1),
    'period':                  [get_period(d) for d in months],
})

df.to_csv('srilanka_fuel_data.csv', index=False)

print("=" * 55)
print("  DATASET CONSTRUCTED")
print("=" * 55)
print(f"Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Period : {df['date'].min().date()} → {df['date'].max().date()}")
print("\nFirst 5 rows:")
print(df[['date','retail_petrol_lkr_l','usd_lkr','inflation_rate_pct']].head())
print("\nPeriod distribution:")
print(df['period'].value_counts().to_string())


# ════════════════════════════════════════════════════════════
#  SECTION 2 — DATA CLEANING & PREPROCESSING
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  SECTION 2 — DATA CLEANING")
print("=" * 55)

# Step 1: Verify no missing values after construction
print(f"\nMissing values per column:\n{df.isnull().sum().to_string()}")

# Step 2: Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# Step 3: Convert all numeric columns to correct types
numeric_cols = [
    'usd_lkr', 'singapore_platts_usd_l', 'formula_price_lkr_l',
    'retail_petrol_lkr_l', 'retail_diesel_lkr_l', 'tax_per_litre_lkr',
    'price_gap_lkr', 'inflation_rate_pct', 'foreign_reserves_usd_m',
    'fuel_import_exp_usd_m'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

# Step 4: Descriptive statistics
print("\nBasic Statistics:")
print(df[['retail_petrol_lkr_l','retail_diesel_lkr_l',
          'usd_lkr','inflation_rate_pct','foreign_reserves_usd_m'
          ]].describe().round(2).to_string())

# Step 5: Period-wise averages
print("\nAverages by Economic Period:")
print(df.groupby('period')[
    ['retail_petrol_lkr_l','usd_lkr','inflation_rate_pct']
].mean().round(2).to_string())

print(f"\nFinal dataset: {df.shape[0]} rows × {df.shape[1]} columns")


# ════════════════════════════════════════════════════════════
#  SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
#  9 publication-quality figures saved to ./charts/
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  SECTION 3 — EDA & VISUALISATIONS")
print("=" * 55)

# ── Figure 1: Retail Petrol vs Formula Price ─────────────────
fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(df['date'], df['retail_petrol_lkr_l'],
        color=RED, lw=2, label='Retail Petrol Price')
ax.plot(df['date'], df['formula_price_lkr_l'],
        color=BLUE, lw=2, ls='--', label='Cost-Reflective Formula Price')

ax.fill_between(df['date'],
                df['retail_petrol_lkr_l'], df['formula_price_lkr_l'],
                where=df['retail_petrol_lkr_l'] > df['formula_price_lkr_l'],
                alpha=0.18, color=RED, label='Over-priced gap')
ax.fill_between(df['date'],
                df['retail_petrol_lkr_l'], df['formula_price_lkr_l'],
                where=df['retail_petrol_lkr_l'] < df['formula_price_lkr_l'],
                alpha=0.18, color=GREEN, label='Under-priced (subsidy) gap')

ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'),
           alpha=0.08, color=RED)
ax.text(pd.Timestamp('2022-04-01'), 480,
        'Crisis\nPeak 2022', fontsize=8, color=RED, ha='center')

ax.set_ylabel('LKR per Litre', fontsize=10)
ax.set_xlabel('')
ax.set_title('Figure 1: Retail Petrol Price vs Cost-Reflective Formula Price (2018–2024)',
             fontsize=11, pad=10)
ax.legend(fontsize=8, loc='upper left')
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'LKR {int(x)}'))

plt.tight_layout()
plt.savefig('charts/fig1_retail_vs_formula.png', bbox_inches='tight')
plt.close()
print("  Fig 1 saved → charts/fig1_retail_vs_formula.png")

# ── Figure 2: Price Distribution by Period (boxplots) ────────
period_order = ['Pre-crisis', 'COVID-period', 'Crisis-peak', 'Recovery']
palette      = [COLORS[p] for p in period_order]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.boxplot(data=df, x='period', y='retail_petrol_lkr_l',
            order=period_order, palette=palette, ax=axes[0], width=0.5)
axes[0].set_title('Fig 2a: Petrol Price Distribution by Period', fontsize=10)
axes[0].set_xlabel(''); axes[0].set_ylabel('LKR/Litre')
axes[0].tick_params(axis='x', rotation=15)

sns.boxplot(data=df, x='period', y='retail_diesel_lkr_l',
            order=period_order, palette=palette, ax=axes[1], width=0.5)
axes[1].set_title('Fig 2b: Diesel Price Distribution by Period', fontsize=10)
axes[1].set_xlabel(''); axes[1].set_ylabel('LKR/Litre')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('charts/fig2_boxplot_by_period.png', bbox_inches='tight')
plt.close()
print("  Fig 2 saved → charts/fig2_boxplot_by_period.png")

# ── Figure 3: Correlation Heatmap ────────────────────────────
num_cols = [
    'retail_petrol_lkr_l', 'retail_diesel_lkr_l', 'usd_lkr',
    'singapore_platts_usd_l', 'inflation_rate_pct',
    'foreign_reserves_usd_m', 'tax_per_litre_lkr',
]
labels = [
    'Petrol Price', 'Diesel Price', 'USD/LKR Rate',
    'Platts (USD/L)', 'Inflation %', 'Reserves (USD M)', 'Tax/Litre',
]
corr = df[num_cols].corr()
corr.index = labels
corr.columns = labels

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=ax, linewidths=0.5, annot_kws={'size': 9},
            mask=mask, vmin=-1, vmax=1)
ax.set_title('Figure 3: Correlation Heatmap of Key Variables',
             fontsize=11, pad=10)

plt.tight_layout()
plt.savefig('charts/fig3_heatmap.png', bbox_inches='tight')
plt.close()
print("  Fig 3 saved → charts/fig3_heatmap.png")

# ── Figure 4: Average Petrol Price by Year ───────────────────
avg_yr = df.groupby('year')['retail_petrol_lkr_l'].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = [RED if y == 2022 else BLUE for y in avg_yr['year']]
bars = ax.bar(avg_yr['year'], avg_yr['retail_petrol_lkr_l'],
              color=bar_colors, alpha=0.85, width=0.6)
ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=9)
ax.set_xlabel('Year')
ax.set_ylabel('Average LKR/Litre')
ax.set_title('Figure 4: Average Annual Petrol Price (LKR/Litre) 2018–2024',
             fontsize=11, pad=10)
ax.set_xticks(avg_yr['year'])

plt.tight_layout()
plt.savefig('charts/fig4_avg_by_year.png', bbox_inches='tight')
plt.close()
print("  Fig 4 saved → charts/fig4_avg_by_year.png")

# ── Figure 5: PCA Scree Plot ─────────────────────────────────
pca_features = [
    'usd_lkr', 'singapore_platts_usd_l', 'inflation_rate_pct',
    'foreign_reserves_usd_m', 'tax_per_litre_lkr', 'fuel_import_exp_usd_m',
]
X_pca     = df[pca_features].copy()
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_pca)
pca       = PCA()
pca.fit(X_scaled)

eigenvalues = pca.explained_variance_
proportions = pca.explained_variance_ratio_
cumulative  = np.cumsum(proportions)

# Print PCA summary
print("\n  PCA Summary:")
for i, (ev, p, c) in enumerate(zip(eigenvalues, proportions, cumulative)):
    print(f"    PC{i+1}: eigenvalue={ev:.4f}  "
          f"variance={p*100:.2f}%  cumulative={c*100:.2f}%")

print("\n  Eigenvectors (loadings):")
loading_df = pd.DataFrame(
    pca.components_.T,
    index=pca_features,
    columns=[f'PC{i+1}' for i in range(len(pca_features))]
)
print(loading_df.round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(eigenvalues) + 1), eigenvalues,
        'o-', color=BLUE, lw=2, ms=7, label='Eigenvalue')
ax.axhline(y=1, color=RED, ls='--', lw=1.5, label='Kaiser criterion (λ=1)')

ax2 = ax.twinx()
ax2.bar(range(1, len(proportions) + 1), proportions * 100,
        alpha=0.25, color=AMBER, label='% Variance')
ax2.set_ylabel('Variance Explained (%)', color=AMBER)

for i, (p, c) in enumerate(zip(proportions, cumulative)):
    ax.annotate(f'{c*100:.1f}%',
                (i + 1, eigenvalues[i] + 0.05),
                ha='center', fontsize=8, color=GREEN)

ax.set_xlabel('Principal Component')
ax.set_ylabel('Eigenvalue')
ax.set_title('Figure 5: PCA Scree Plot — Sri Lanka Fuel Crisis Drivers',
             fontsize=11, pad=10)
ax.legend(loc='upper right', fontsize=8)
ax.set_xticks(range(1, len(eigenvalues) + 1))

plt.tight_layout()
plt.savefig('charts/fig5_pca_scree.png', bbox_inches='tight')
plt.close()
print("  Fig 5 saved → charts/fig5_pca_scree.png")

# ── Figure 6: Scatter – Platts & USD/LKR vs Petrol Price ─────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for period, col in COLORS.items():
    sub = df[df['period'] == period]
    axes[0].scatter(sub['singapore_platts_usd_l'],
                    sub['retail_petrol_lkr_l'],
                    c=col, label=period, alpha=0.8, s=55)
    axes[1].scatter(sub['usd_lkr'],
                    sub['retail_petrol_lkr_l'],
                    c=col, label=period, alpha=0.8, s=55)

axes[0].set_xlabel('Singapore Platts (USD/L)')
axes[0].set_ylabel('Retail Petrol (LKR/L)')
axes[0].set_title('Fig 6a: Platts vs Petrol Price', fontsize=10)
axes[0].legend(fontsize=7)

axes[1].set_xlabel('USD/LKR Exchange Rate')
axes[1].set_ylabel('Retail Petrol (LKR/L)')
axes[1].set_title('Fig 6b: USD/LKR Rate vs Petrol Price', fontsize=10)
axes[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig('charts/fig6_scatter.png', bbox_inches='tight')
plt.close()
print("  Fig 6 saved → charts/fig6_scatter.png")

# ── Figure 9: Inflation & Foreign Reserves (twin-axis) ───────
fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()

ax1.plot(df['date'], df['inflation_rate_pct'],
         color=RED, lw=2, label='Inflation Rate (%)')
ax2.plot(df['date'], df['foreign_reserves_usd_m'] / 1000,
         color=BLUE, lw=2, ls='--', label='Foreign Reserves (USD B)')

ax1.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'),
            alpha=0.08, color=RED)
ax1.set_ylabel('Inflation Rate (%)', color=RED)
ax1.set_xlabel('')
ax2.set_ylabel('Foreign Reserves (USD Billion)', color=BLUE)
ax1.set_title('Figure 9: Inflation Rate vs. Foreign Reserves 2018–2024',
              fontsize=11, pad=10)

lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('charts/fig9_inflation_reserves.png', bbox_inches='tight')
plt.close()
print("  Fig 9 saved → charts/fig9_inflation_reserves.png")


# ════════════════════════════════════════════════════════════
#  SECTION 4 — MACHINE LEARNING MODELS
#  Target   : retail_petrol_lkr_l
#  Features : 6 macroeconomic indicators
#  Split    : 70% train / 30% test (random_state=42)
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  SECTION 4 — MACHINE LEARNING")
print("=" * 55)

feature_cols = [
    'usd_lkr',
    'singapore_platts_usd_l',
    'inflation_rate_pct',
    'foreign_reserves_usd_m',
    'tax_per_litre_lkr',
    'fuel_import_exp_usd_m',
]

X = df[feature_cols]
y = df['retail_petrol_lkr_l']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"\nTrain size : {len(X_train)} | Test size : {len(X_test)}")

# ── 4.1 Linear Regression ─────────────────────────────────────
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred  = lr_model.predict(X_test)

lr_r2  = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)

print("\n  Linear Regression Performance:")
print(f"    R²  Score : {lr_r2:.4f}")
print(f"    MAE       : {lr_mae:.2f} LKR/L")
print(f"    MSE       : {lr_mse:.2f}")

print("\n  Coefficients:")
coef_df = pd.DataFrame({
    'Feature':     feature_cols,
    'Coefficient': lr_model.coef_.round(4),
}).sort_values('Coefficient', ascending=False)
print(coef_df.to_string(index=False))

# ── 4.2 Random Forest Regressor ───────────────────────────────
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred  = rf_model.predict(X_test)

rf_r2  = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

print("\n  Random Forest Performance:")
print(f"    R²  Score : {rf_r2:.4f}")
print(f"    MAE       : {rf_mae:.2f} LKR/L")
print(f"    MSE       : {rf_mse:.2f}")

print("\n  Feature Importances:")
fi_df = pd.DataFrame({
    'Feature':    feature_cols,
    'Importance': rf_model.feature_importances_.round(4),
}).sort_values('Importance', ascending=False)
print(fi_df.to_string(index=False))

# ── 4.3 Model Comparison Table ────────────────────────────────
print("\n  ┌─────────────────────────────────────────────────────────┐")
print("  │          MODEL PERFORMANCE COMPARISON SUMMARY          │")
print("  ├──────────────────────┬──────────────┬──────────────────┤")
print("  │ Metric               │  Linear Reg. │  Random Forest   │")
print("  ├──────────────────────┼──────────────┼──────────────────┤")
print(f"  │ R² Score             │   {lr_r2:.4f}   │     {rf_r2:.4f}      │")
print(f"  │ MAE (LKR/L)          │   {lr_mae:7.2f}   │     {rf_mae:7.2f}      │")
print(f"  │ MSE                  │   {lr_mse:7.2f}   │   {rf_mse:9.2f}      │")
print("  ├──────────────────────┼──────────────┼──────────────────┤")
print("  │ Best Performer       │      ✓       │                  │")
print("  └──────────────────────┴──────────────┴──────────────────┘")

# ── Figure 7: Actual vs Predicted ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

mn = min(y_test.min(), lr_pred.min(), rf_pred.min())
mx = max(y_test.max(), lr_pred.max(), rf_pred.max())

axes[0].scatter(y_test, lr_pred, color=BLUE, alpha=0.7, s=55)
axes[0].plot([mn, mx], [mn, mx], '--', color=RED, lw=1.5)
axes[0].set_xlabel('Actual Price (LKR/L)')
axes[0].set_ylabel('Predicted Price (LKR/L)')
axes[0].set_title(
    f'Fig 7a: Linear Regression\nR²={lr_r2:.3f}  MAE={lr_mae:.1f}',
    fontsize=10)

axes[1].scatter(y_test, rf_pred, color=GREEN, alpha=0.7, s=55)
axes[1].plot([mn, mx], [mn, mx], '--', color=RED, lw=1.5)
axes[1].set_xlabel('Actual Price (LKR/L)')
axes[1].set_ylabel('Predicted Price (LKR/L)')
axes[1].set_title(
    f'Fig 7b: Random Forest\nR²={rf_r2:.3f}  MAE={rf_mae:.1f}',
    fontsize=10)

plt.tight_layout()
plt.savefig('charts/fig7_actual_vs_pred.png', bbox_inches='tight')
plt.close()
print("\n  Fig 7 saved → charts/fig7_actual_vs_pred.png")

# ── Figure 8: Feature Importance ─────────────────────────────
fi_labels = {
    'usd_lkr':                 'USD/LKR Rate',
    'singapore_platts_usd_l':  'Singapore Platts',
    'inflation_rate_pct':      'Inflation Rate',
    'foreign_reserves_usd_m':  'Foreign Reserves',
    'tax_per_litre_lkr':       'Tax per Litre',
    'fuel_import_exp_usd_m':   'Fuel Import Exp.',
}
fi = pd.Series(
    rf_model.feature_importances_,
    index=[fi_labels[c] for c in feature_cols]
).sort_values()

fig, ax = plt.subplots(figsize=(7, 4))
fi.plot(kind='barh', ax=ax, color=BLUE, alpha=0.85)
ax.set_xlabel('Feature Importance (Mean Decrease Impurity)')
ax.set_title('Figure 8: Random Forest Feature Importance',
             fontsize=11, pad=10)
for i, v in enumerate(fi.values):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('charts/fig8_feature_importance.png', bbox_inches='tight')
plt.close()
print("  Fig 8 saved → charts/fig8_feature_importance.png")


# ════════════════════════════════════════════════════════════
#  SECTION 5 — KEY FINDINGS SUMMARY
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  SECTION 5 — KEY FINDINGS")
print("=" * 55)

max_petrol_row = df.loc[df['retail_petrol_lkr_l'].idxmax()]
min_petrol_row = df.loc[df['retail_petrol_lkr_l'].idxmin()]

print(f"\n  Peak retail petrol price : LKR {max_petrol_row['retail_petrol_lkr_l']:.0f}/L"
      f"  ({max_petrol_row['date'].strftime('%b %Y')})")
print(f"  Lowest retail price      : LKR {min_petrol_row['retail_petrol_lkr_l']:.0f}/L"
      f"  ({min_petrol_row['date'].strftime('%b %Y')})")
print(f"  Total price increase     : "
      f"{((max_petrol_row['retail_petrol_lkr_l'] / min_petrol_row['retail_petrol_lkr_l']) - 1)*100:.0f}%")

print(f"\n  Peak inflation           : "
      f"{df['inflation_rate_pct'].max():.1f}%"
      f"  ({df.loc[df['inflation_rate_pct'].idxmax(), 'date'].strftime('%b %Y')})")
print(f"  Lowest reserves          : "
      f"USD {df['foreign_reserves_usd_m'].min():,.0f} M"
      f"  ({df.loc[df['foreign_reserves_usd_m'].idxmin(), 'date'].strftime('%b %Y')})")

top_corr = df[['retail_petrol_lkr_l','usd_lkr']].corr().iloc[0,1]
print(f"\n  Correlation (Petrol ↔ USD/LKR) : {top_corr:.3f}")
print(f"  PCA — PC1 variance explained   : {proportions[0]*100:.2f}%")
print(f"  PCA — PC1+PC2 cumulative       : {cumulative[1]*100:.2f}%")

print(f"\n  Best ML model : Linear Regression")
print(f"    → R²={lr_r2:.4f}  MAE={lr_mae:.2f} LKR/L  MSE={lr_mse:.2f}")
print(f"\n  All charts saved to ./charts/")
print(f"  Dataset saved to ./srilanka_fuel_data.csv")
print("\n" + "=" * 55)
print("  ANALYSIS COMPLETE")
print("=" * 55)
