import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('../data/processed/engineered.csv')
df = df.sort_values('date')
df['rest_category_encoded'] = df['rest_category'].astype('category')

# Define features
features = [
    'eFG_pct_avg_last_20', 'tov_rate_avg_last_20', 'oreb_pct_avg_last_20', 'ftr_avg_last_20',
    'ortg_avg_last_20', 'drtg_avg_last_20', 'covered_avg_last_20', 
    'eFG_pct_avg_last_5', 'tov_rate_avg_last_5', 'oreb_pct_avg_last_5', 'ftr_avg_last_5',
    'ortg_avg_last_5', 'drtg_avg_last_5', 'covered_avg_last_5', 
    # 'days_of_rest', 
    # 'opp_days_of_rest',
    'home_team', 
    'opp_ortg_avg_last_20', 'opp_drtg_avg_last_20',
    'opp_ortg_avg_last_5', 'opp_drtg_avg_last_5',
    'rest_category_encoded'
]

target = 'covered'


X = df[features]
y = df['covered']

# Train model using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    enable_categorical=True,
    tree_method='hist',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Get predictions on test set
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, (preds > 0.5).astype(int))
    print(f"AUC: {auc:.3f} | Accuracy: {acc:.3f}")

# Store predictions
df.loc[test_idx, 'pred_prob'] = preds
df = df.sort_values('date')

# Betting parameters
upper_thresh = 0.6
WIN_PAYOUT = 0.9091
LOSS_PAYOUT = -1.0

# Identify bets
df['bet_side'] = np.where(df['pred_prob'] > upper_thresh, 1, np.nan)
bets = df.dropna(subset=['bet_side']).copy()
bets['bet_outcome'] = np.where(
    (bets['bet_side'] == bets['covered']), WIN_PAYOUT, LOSS_PAYOUT
)

print(f"\nTotal bets placed: {len(bets)}")
print(f"Actual total units won: {bets['bet_outcome'].sum():.2f}")

# ===== MONTE CARLO SIMULATION =====
n_simulations = 10000
n_bets = 10000
bet_outcomes = bets['bet_outcome'].values

# Storage for results
simulated_totals = np.zeros(n_simulations)

np.random.seed(42)
for i in range(n_simulations):
    # Sample with replacement from actual bet outcomes
    sampled_outcomes = np.random.choice(bet_outcomes, size=n_bets, replace=True)
    simulated_totals[i] = sampled_outcomes.sum()

# Calculate statistics
mean_units = simulated_totals.mean()
median_units = np.median(simulated_totals)
std_units = simulated_totals.std()
ci_lower = np.percentile(simulated_totals, 2.5)
ci_upper = np.percentile(simulated_totals, 97.5)

print(f"\n{'='*50}")
print(f"Monte Carlo Simulation Results ({n_simulations:,} simulations)")
print(f"{'='*50}")
print(f"Mean total units won: {mean_units:.2f}")
print(f"Median total units won: {median_units:.2f}")
print(f"Std deviation: {std_units:.2f}")
print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Probability of profit: {(simulated_totals > 0).mean():.2%}")
print(f"Best case (95th percentile): {np.percentile(simulated_totals, 95):.2f}")
print(f"Worst case (5th percentile): {np.percentile(simulated_totals, 5):.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(simulated_totals, bins=40, alpha=0.7, edgecolor='black')
axes[0].axvline(mean_units, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_units:.2f}')
axes[0].axvline(ci_lower, color='orange', linestyle='--', linewidth=1.5, label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
axes[0].axvline(ci_upper, color='orange', linestyle='--', linewidth=1.5)
axes[0].axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
axes[0].set_xlabel('Total Units Won')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Total Units Won\n(Monte Carlo Simulation)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative distribution
sorted_totals = np.sort(simulated_totals)
cumulative_prob = np.arange(1, len(sorted_totals) + 1) / len(sorted_totals)
axes[1].plot(sorted_totals, cumulative_prob, linewidth=2)
axes[1].axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Break-even')
axes[1].axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1].axvline(median_units, color='red', linestyle='--', linewidth=2, label=f'Median: {median_units:.2f}')
axes[1].set_xlabel('Total Units Won')
axes[1].set_ylabel('Cumulative Probability')
axes[1].set_title('Cumulative Distribution Function')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('monte_carlo_results.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'monte_carlo_results.png'")
# plt.show()  # Commented out for non-interactive backend

# Risk metrics
print(f"\n{'='*50}")
print(f"Risk Metrics")
print(f"{'='*50}")
print(f"Value at Risk (VaR 95%): {np.percentile(simulated_totals, 5):.2f}")
print(f"Sharpe-like ratio: {mean_units / std_units:.3f}")
print(f"Max simulated profit: {simulated_totals.max():.2f}")
print(f"Max simulated loss: {simulated_totals.min():.2f}")