import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Configuration du style moderne
plt.style.use('dark_background')
sns.set_palette("husl")

# Charger les données
df = pd.read_csv('Data/SP500_features.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Sélection des features pertinentes (enlever les NaN)
df_clean = df.dropna()

print("🎨 Génération des graphiques en cours...\n")

# ==============================================================================
# GRAPHIQUE 1: SCORE CLOSE + INDICATEURS TECHNIQUES
# ==============================================================================
print("📊 1/7 - Score Close avec indicateurs techniques...")
fig1 = plt.figure(figsize=(18, 12))
gs = fig1.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)

# Subplot 1: Score Close avec EMA et Bollinger Bands
ax1 = fig1.add_subplot(gs[0])
ax1.plot(df_clean['Date'], df_clean['Close'], label='S&P 500 Score', 
         linewidth=2, color='#00d4ff', alpha=0.9)
ax1.plot(df_clean['Date'], df_clean['EMA_12'], label='EMA 12', 
         linewidth=1.5, color='#ff0051', alpha=0.8, linestyle='--')
ax1.plot(df_clean['Date'], df_clean['EMA_26'], label='EMA 26', 
         linewidth=1.5, color='#00ff88', alpha=0.8, linestyle='--')

# Bollinger Bands
ax1.fill_between(df_clean['Date'], df_clean['BB_Upper'], df_clean['BB_Lower'], 
                  alpha=0.2, color='#ffaa00', label='Bollinger Bands')
ax1.plot(df_clean['Date'], df_clean['BB_Mid'], linewidth=1, 
         color='#ffaa00', alpha=0.6, linestyle=':')

ax1.set_ylabel('Score S&P 500', fontsize=12, fontweight='bold')
ax1.set_title('Score S&P 500 avec Indicateurs Techniques', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax1.grid(alpha=0.3)
ax1.set_xlim(df_clean['Date'].min(), df_clean['Date'].max())

# Subplot 2: RSI
ax2 = fig1.add_subplot(gs[1], sharex=ax1)
ax2.plot(df_clean['Date'], df_clean['RSI_14'], linewidth=2, color='#ff00ff', alpha=0.9)
ax2.axhline(70, color='#ff0051', linestyle='--', linewidth=1.5, alpha=0.7, label='Sur-achat (70)')
ax2.axhline(30, color='#00ff88', linestyle='--', linewidth=1.5, alpha=0.7, label='Survente (30)')
ax2.axhline(50, color='white', linestyle=':', linewidth=1, alpha=0.5)
ax2.fill_between(df_clean['Date'], 70, 100, alpha=0.15, color='#ff0051')
ax2.fill_between(df_clean['Date'], 0, 30, alpha=0.15, color='#00ff88')
ax2.set_ylabel('RSI (14)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(alpha=0.3)

# Subplot 3: MACD
ax3 = fig1.add_subplot(gs[2], sharex=ax1)
ax3.plot(df_clean['Date'], df_clean['MACD'], linewidth=2, color='#00d4ff', 
         alpha=0.9, label='MACD')
ax3.plot(df_clean['Date'], df_clean['MACD_signal'], linewidth=2, color='#ff0051', 
         alpha=0.9, label='Signal')
ax3.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
# Histogramme MACD
macd_hist = df_clean['MACD'] - df_clean['MACD_signal']
colors = ['#00ff88' if x > 0 else '#ff0051' for x in macd_hist]
ax3.bar(df_clean['Date'], macd_hist, width=1, alpha=0.3, color=colors)
ax3.set_ylabel('MACD', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(alpha=0.3)

# Subplot 4: Volume
ax4 = fig1.add_subplot(gs[3], sharex=ax1)
ax4.bar(df_clean['Date'], df_clean['Volume'], width=1, alpha=0.6, color='#ffaa00')
ax4.plot(df_clean['Date'], df_clean['Volume_SMA20'], linewidth=2, 
         color='#ff0051', alpha=0.9, label='SMA Volume (20)')
ax4.set_ylabel('Volume', fontsize=11, fontweight='bold')
ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(alpha=0.3)
ax4.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig('1_score_close_analysis.png', dpi=300, bbox_inches='tight', facecolor='#0e1117')
plt.close()

# ==============================================================================
# GRAPHIQUE 2: RSI DÉTAILLÉ
# ==============================================================================
print("📈 2/7 - RSI détaillé...")
fig2, ax = plt.subplots(figsize=(18, 8))

ax.plot(df_clean['Date'], df_clean['RSI_14'], linewidth=2.5, color='#ff00ff', alpha=0.9, label='RSI (14)')
ax.axhline(70, color='#ff0051', linestyle='--', linewidth=2, alpha=0.8, label='Sur-achat (70)')
ax.axhline(30, color='#00ff88', linestyle='--', linewidth=2, alpha=0.8, label='Survente (30)')
ax.axhline(50, color='white', linestyle=':', linewidth=1.5, alpha=0.6, label='Neutre (50)')

# Zones colorées
ax.fill_between(df_clean['Date'], 70, 100, alpha=0.2, color='#ff0051', label='Zone sur-achat')
ax.fill_between(df_clean['Date'], 0, 30, alpha=0.2, color='#00ff88', label='Zone survente')
ax.fill_between(df_clean['Date'], 30, 70, alpha=0.1, color='gray', label='Zone neutre')

ax.set_ylabel('RSI', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_title('Relative Strength Index (RSI) - Analyse Complète', fontsize=18, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3)

# Statistiques RSI
rsi_mean = df_clean['RSI_14'].mean()
rsi_current = df_clean['RSI_14'].iloc[-1]
ax.text(0.02, 0.98, f'RSI actuel: {rsi_current:.2f}\nRSI moyen: {rsi_mean:.2f}', 
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

plt.tight_layout()
plt.savefig('2_rsi_detailed.png', dpi=300, bbox_inches='tight', facecolor='#0e1117')
plt.close()

# ==============================================================================
# GRAPHIQUE 3: MOMENTUM (5, 10, 20 jours)
# ==============================================================================
print("🚀 3/7 - Analyse des Momentum...")
fig3, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

# Momentum 5
axes[0].plot(df_clean['Date'], df_clean['Momentum_5'], linewidth=2.5, 
             color='#00d4ff', alpha=0.9, label='Momentum 5 jours')
axes[0].axhline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.6)
axes[0].fill_between(df_clean['Date'], df_clean['Momentum_5'], 0, 
                      where=(df_clean['Momentum_5'] > 0), alpha=0.3, color='#00ff88', interpolate=True)
axes[0].fill_between(df_clean['Date'], df_clean['Momentum_5'], 0, 
                      where=(df_clean['Momentum_5'] <= 0), alpha=0.3, color='#ff0051', interpolate=True)
axes[0].set_ylabel('Momentum 5', fontsize=12, fontweight='bold')
axes[0].set_title('Momentum Court Terme (5 jours)', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=11)
axes[0].grid(alpha=0.3)

# Momentum 10
axes[1].plot(df_clean['Date'], df_clean['Momentum_10'], linewidth=2.5, 
             color='#ff00ff', alpha=0.9, label='Momentum 10 jours')
axes[1].axhline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.6)
axes[1].fill_between(df_clean['Date'], df_clean['Momentum_10'], 0, 
                      where=(df_clean['Momentum_10'] > 0), alpha=0.3, color='#00ff88', interpolate=True)
axes[1].fill_between(df_clean['Date'], df_clean['Momentum_10'], 0, 
                      where=(df_clean['Momentum_10'] <= 0), alpha=0.3, color='#ff0051', interpolate=True)
axes[1].set_ylabel('Momentum 10', fontsize=12, fontweight='bold')
axes[1].set_title('Momentum Moyen Terme (10 jours)', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper left', fontsize=11)
axes[1].grid(alpha=0.3)

# Momentum 20
axes[2].plot(df_clean['Date'], df_clean['Momentum_20'], linewidth=2.5, 
             color='#ffaa00', alpha=0.9, label='Momentum 20 jours')
axes[2].axhline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.6)
axes[2].fill_between(df_clean['Date'], df_clean['Momentum_20'], 0, 
                      where=(df_clean['Momentum_20'] > 0), alpha=0.3, color='#00ff88', interpolate=True)
axes[2].fill_between(df_clean['Date'], df_clean['Momentum_20'], 0, 
                      where=(df_clean['Momentum_20'] <= 0), alpha=0.3, color='#ff0051', interpolate=True)
axes[2].set_ylabel('Momentum 20', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Date', fontsize=14, fontweight='bold')
axes[2].set_title('Momentum Long Terme (20 jours)', fontsize=14, fontweight='bold')
axes[2].legend(loc='upper left', fontsize=11)
axes[2].grid(alpha=0.3)

fig3.suptitle('Analyse Complète des Momentum', fontsize=18, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig('3_momentum_analysis.png', dpi=300, bbox_inches='tight', facecolor='#0e1117')
plt.close()

# ==============================================================================
# GRAPHIQUE 4: VOLATILITÉ (ATR + Volatility)
# ==============================================================================
print("📉 4/7 - Analyse de la Volatilité...")
fig4, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

# ATR
axes[0].plot(df_clean['Date'], df_clean['ATR_14'], linewidth=2.5, 
             color='#ff0051', alpha=0.9, label='ATR (14)')
axes[0].fill_between(df_clean['Date'], df_clean['ATR_14'], alpha=0.3, color='#ff0051')
axes[0].set_ylabel('ATR (14)', fontsize=12, fontweight='bold')
axes[0].set_title('Average True Range - Mesure de la Volatilité', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=11)
axes[0].grid(alpha=0.3)

# Statistiques ATR
atr_mean = df_clean['ATR_14'].mean()
atr_current = df_clean['ATR_14'].iloc[-1]
axes[0].axhline(atr_mean, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label=f'ATR moyen: {atr_mean:.2f}')
axes[0].text(0.02, 0.98, f'ATR actuel: {atr_current:.2f}\nATR moyen: {atr_mean:.2f}', 
             transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

# Volatility 20
axes[1].plot(df_clean['Date'], df_clean['Volatility_20'], linewidth=2.5, 
             color='#00d4ff', alpha=0.9, label='Volatility (20)')
axes[1].fill_between(df_clean['Date'], df_clean['Volatility_20'], alpha=0.3, color='#00d4ff')
axes[1].set_ylabel('Volatility (20)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=14, fontweight='bold')
axes[1].set_title('Volatilité sur 20 jours', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper left', fontsize=11)
axes[1].grid(alpha=0.3)

# Statistiques Volatility
vol_mean = df_clean['Volatility_20'].mean()
vol_current = df_clean['Volatility_20'].iloc[-1]
axes[1].axhline(vol_mean, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Vol moyen: {vol_mean:.4f}')
axes[1].text(0.02, 0.98, f'Vol actuelle: {vol_current:.4f}\nVol moyenne: {vol_mean:.4f}', 
             transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

fig4.suptitle('Analyse de la Volatilité', fontsize=18, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig('4_volatility_analysis.png', dpi=300, bbox_inches='tight', facecolor='#0e1117')
plt.close()

# ==============================================================================
# GRAPHIQUE 5: MATRICE DE CORRÉLATION
# ==============================================================================
print("🔥 5/7 - Matrice de corrélation...")
fig5, ax5 = plt.subplots(figsize=(16, 12))

features = ['Return_1', 'LogReturn_1', 'Momentum_5', 'Momentum_10', 'Momentum_20', 
            'RSI_14', 'MACD', 'MACD_signal', 'Volatility_20', 'ATR_14', 
            'Volume_Normalized', 'H-L', 'TrueRange']

corr_matrix = df_clean[features].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1.5, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax5, annot_kws={'size': 10})

ax5.set_title('Matrice de Corrélation - Features vs Return_1', 
              fontsize=18, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig('5_correlation_matrix.png', dpi=300, bbox_inches='tight', facecolor='#0e1117')
plt.close()

# ==============================================================================
# GRAPHIQUE 6: DISTRIBUTION DES RENDEMENTS + Q-Q PLOT
# ==============================================================================
print("📊 6/7 - Distribution des rendements...")
fig6, axes = plt.subplots(1, 2, figsize=(18, 8))

returns = df_clean['Return_1'].dropna()

# Histogramme avec KDE
axes[0].hist(returns, bins=100, alpha=0.7, color='#00d4ff', edgecolor='white', density=True)
axes[0].axvline(returns.mean(), color='#ff0051', linestyle='--', linewidth=2.5, 
                label=f'Moyenne: {returns.mean():.6f}')
axes[0].axvline(returns.median(), color='#00ff88', linestyle='--', linewidth=2.5, 
                label=f'Médiane: {returns.median():.6f}')

# KDE overlay
from scipy.stats import gaussian_kde
kde = gaussian_kde(returns)
x_range = np.linspace(returns.min(), returns.max(), 1000)
axes[0].plot(x_range, kde(x_range), color='yellow', linewidth=3, label='KDE')

axes[0].set_xlabel('Return_1', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Densité', fontsize=13, fontweight='bold')
axes[0].set_title('Distribution des Rendements', fontsize=16, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Statistiques
skew = stats.skew(returns)
kurt = stats.kurtosis(returns)
axes[0].text(0.02, 0.98, f'Skewness: {skew:.4f}\nKurtosis: {kurt:.4f}\nStd: {returns.std():.6f}', 
             transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

# Q-Q Plot
stats.probplot(returns, dist="norm", plot=axes[1])
axes[1].get_lines()[0].set_color('#00d4ff')
axes[1].get_lines()[0].set_markersize(5)
axes[1].get_lines()[1].set_color('#ff0051')
axes[1].get_lines()[1].set_linewidth(3)
axes[1].set_title('Q-Q Plot - Test de Normalité', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Quantiles Théoriques', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Quantiles Observés', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)

fig6.suptitle('Analyse de la Distribution des Rendements', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('6_distribution_returns.png', dpi=300, bbox_inches='tight', facecolor='#0e1117')
plt.close()

# ==============================================================================
# GRAPHIQUE 7: AUTOCORRÉLATION (ACF + PACF)
# ==============================================================================
print("🎯 7/7 - Autocorrélation (ACF/PACF)...")
fig7, axes = plt.subplots(2, 1, figsize=(18, 12))

returns_clean = df_clean['Return_1'].dropna()

# ACF
plot_acf(returns_clean, lags=50, ax=axes[0], color='#00d4ff', alpha=0.8)
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('Lag', fontsize=13, fontweight='bold')
axes[0].set_ylabel('ACF', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

# PACF
plot_pacf(returns_clean, lags=50, ax=axes[1], color='#ff0051', alpha=0.8)
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('Lag', fontsize=13, fontweight='bold')
axes[1].set_ylabel('PACF', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

fig7.suptitle('Analyse d\'Autocorrélation - Détermination des Lags Optimaux', 
              fontsize=18, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig('7_autocorrelation.png', dpi=300, bbox_inches='tight', facecolor='#0e1117')
plt.close()

# ==============================================================================
# RÉSUMÉ STATISTIQUE
# ==============================================================================
print("\n" + "="*80)
print("📊 RÉSUMÉ DE L'ANALYSE S&P 500")
print("="*80)

print(f"\n📅 Période analysée: {df_clean['Date'].min().date()} à {df_clean['Date'].max().date()}")
print(f"📈 Nombre de jours: {len(df_clean)}")
print(f"\n💰 SCORE S&P 500:")
print(f"  Actuel: {df_clean['Close'].iloc[-1]:.2f}")
print(f"  Minimum: {df_clean['Close'].min():.2f} ({df_clean[df_clean['Close'] == df_clean['Close'].min()]['Date'].iloc[0].date()})")
print(f"  Maximum: {df_clean['Close'].max():.2f} ({df_clean[df_clean['Close'] == df_clean['Close'].max()]['Date'].iloc[0].date()})")
print(f"  Variation totale: {((df_clean['Close'].iloc[-1] - df_clean['Close'].iloc[0]) / df_clean['Close'].iloc[0] * 100):.2f}%")

print("\n💹 RENDEMENTS:")
print(f"  Moyenne: {returns.mean():.6f} ({returns.mean()*100:.4f}% par jour)")
print(f"  Médiane: {returns.median():.6f}")
print(f"  Écart-type: {returns.std():.6f}")
print(f"  Skewness: {skew:.4f} {'(asymétrie négative)' if skew < 0 else '(asymétrie positive)'}")
print(f"  Kurtosis: {kurt:.4f} {'(queues épaisses)' if kurt > 0 else '(queues fines)'}")
print(f"  Min: {returns.min():.6f} ({returns.min()*100:.4f}%)")
print(f"  Max: {returns.max():.6f} ({returns.max()*100:.4f}%)")

print("\n📉 RSI (14):")
rsi_current = df_clean['RSI_14'].iloc[-1]
rsi_mean = df_clean['RSI_14'].mean()
print(f"  Valeur actuelle: {rsi_current:.2f}")
print(f"  Moyenne: {rsi_mean:.2f}")
if rsi_current > 70:
    print("  ⚠️  SIGNAL: SUR-ACHAT")
elif rsi_current < 30:
    print("  ⚠️  SIGNAL: SURVENTE")
else:
    print("  ✓ ZONE NEUTRE")

print("\n🚀 MOMENTUM:")
print(f"  Momentum 5j: {df_clean['Momentum_5'].iloc[-1]:.6f}")
print(f"  Momentum 10j: {df_clean['Momentum_10'].iloc[-1]:.6f}")
print(f"  Momentum 20j: {df_clean['Momentum_20'].iloc[-1]:.6f}")

print("\n📊 VOLATILITÉ:")
print(f"  ATR (14): {df_clean['ATR_14'].iloc[-1]:.2f} (moyenne: {df_clean['ATR_14'].mean():.2f})")
print(f"  Volatility (20): {df_clean['Volatility_20'].iloc[-1]:.6f} (moyenne: {df_clean['Volatility_20'].mean():.6f})")

print("\n🎯 TOP 5 CORRÉLATIONS AVEC RETURN_1:")
corr_with_target = corr_matrix['Return_1'].drop('Return_1').abs().sort_values(ascending=False)
for i, (feat, corr_val) in enumerate(corr_with_target.head(5).items(), 1):
    original_corr = corr_matrix['Return_1'][feat]
    print(f"  {i}. {feat}: {original_corr:+.4f}")

print("\n" + "="*80)
print("✅ 7 graphiques générés avec succès!")
print("="*80)
print("\n📁 Fichiers créés:")
print("  1. 1_score_close_analysis.png")
print("  2. 2_rsi_detailed.png")
print("  3. 3_momentum_analysis.png")
print("  4. 4_volatility_analysis.png")
print("  5. 5_correlation_matrix.png")
print("  6. 6_distribution_returns.png")
print("  7. 7_autocorrelation.png")
print("="*80)