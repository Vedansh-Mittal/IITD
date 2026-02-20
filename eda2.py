import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATA = "data/"

# ============================================================
# RELOAD DATA (run this file fresh, it's standalone)
# ============================================================

print("Loading data...")
customers    = pd.read_csv(f"{DATA}customers.csv")
accounts     = pd.read_csv(f"{DATA}accounts.csv")
linkage      = pd.read_csv(f"{DATA}customer_account_linkage.csv")
products     = pd.read_csv(f"{DATA}product_details.csv")
labels       = pd.read_csv(f"{DATA}train_labels.csv")

transactions = pd.concat(
    [pd.read_csv(f"{DATA}transactions_part_{i}.csv") for i in range(6)],
    ignore_index=True
)
transactions['transaction_timestamp'] = pd.to_datetime(transactions['transaction_timestamp'])

# Build master table
train = labels.copy()
train = train.merge(accounts,  on="account_id",  how="left")
train = train.merge(linkage,   on="account_id",  how="left")
train = train.merge(customers, on="customer_id", how="left")
train = train.merge(products,  on="customer_id", how="left")

# Basic transaction features
txn_features = transactions.groupby('account_id').agg(
    total_txn_count       = ('transaction_id', 'count'),
    total_amount          = ('amount', 'sum'),
    avg_txn_amount        = ('amount', 'mean'),
    max_txn_amount        = ('amount', 'max'),
    unique_counterparties = ('counterparty_id', 'nunique'),
    unique_channels       = ('channel', 'nunique'),
    credit_count          = ('txn_type', lambda x: (x == 'C').sum()),
    debit_count           = ('txn_type', lambda x: (x == 'D').sum()),
).reset_index()
txn_features['credit_debit_ratio'] = txn_features['credit_count'] / (txn_features['debit_count'] + 1)
train = train.merge(txn_features, on='account_id', how='left')

print("Data ready!\n")

# ============================================================
# ANALYSIS 1: ACCOUNT AGE
# How old is the account when flagged? New accounts = suspicious
# ============================================================

print("=" * 60)
print("ANALYSIS 1: ACCOUNT AGE")
print("=" * 60)

train['account_opening_date'] = pd.to_datetime(train['account_opening_date'])
reference_date = pd.Timestamp("2025-06-30")  # end of dataset window
train['account_age_days'] = (reference_date - train['account_opening_date']).dt.days

mule_age  = train[train['is_mule']==1]['account_age_days'].median()
legit_age = train[train['is_mule']==0]['account_age_days'].median()
print(f"Mule median account age:  {mule_age:.0f} days ({mule_age/365:.1f} years)")
print(f"Legit median account age: {legit_age:.0f} days ({legit_age/365:.1f} years)")

# New accounts (< 1 year): what % are mules?
train['is_new_account'] = train['account_age_days'] < 365
new_acct = train[train['is_new_account']]
print(f"\nAmong NEW accounts (<1yr): mule rate = {new_acct['is_mule'].mean():.2%}")
print(f"Among OLD accounts (>1yr): mule rate = {train[~train['is_new_account']]['is_mule'].mean():.2%}")

# ============================================================
# ANALYSIS 2: CUSTOMER AGE (demographics)
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS 2: CUSTOMER AGE")
print("=" * 60)

train['date_of_birth'] = pd.to_datetime(train['date_of_birth'], errors='coerce')
train['customer_age'] = (reference_date - train['date_of_birth']).dt.days / 365

mule_cage  = train[train['is_mule']==1]['customer_age'].median()
legit_cage = train[train['is_mule']==0]['customer_age'].median()
print(f"Mule median customer age:  {mule_cage:.1f} years")
print(f"Legit median customer age: {legit_cage:.1f} years")

# ============================================================
# ANALYSIS 3: RELATIONSHIP TENURE
# How long has the customer been with the bank?
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS 3: RELATIONSHIP TENURE")
print("=" * 60)

train['relationship_start_date'] = pd.to_datetime(train['relationship_start_date'], errors='coerce')
train['tenure_days'] = (reference_date - train['relationship_start_date']).dt.days

mule_ten  = train[train['is_mule']==1]['tenure_days'].median()
legit_ten = train[train['is_mule']==0]['tenure_days'].median()
print(f"Mule median tenure:  {mule_ten:.0f} days ({mule_ten/365:.1f} years)")
print(f"Legit median tenure: {legit_ten:.0f} days ({legit_ten/365:.1f} years)")

# ============================================================
# ANALYSIS 4: RAPID PASS-THROUGH DETECTION
# Mules receive money and immediately send it out
# Key signal: time between credit and next debit is very short
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS 4: RAPID PASS-THROUGH")
print("=" * 60)

# For each account, compute: what % of days had BOTH a credit and debit?
txn_sorted = transactions.sort_values(['account_id', 'transaction_timestamp'])
txn_sorted['date'] = txn_sorted['transaction_timestamp'].dt.date

daily = txn_sorted.groupby(['account_id', 'date', 'txn_type']).size().unstack(fill_value=0)
daily.columns.name = None
if 'C' not in daily.columns: daily['C'] = 0
if 'D' not in daily.columns: daily['D'] = 0

# Days with BOTH credit and debit = pass-through days
daily['both'] = (daily['C'] > 0) & (daily['D'] > 0)
passthrough = daily.groupby('account_id')['both'].mean().reset_index()
passthrough.columns = ['account_id', 'passthrough_rate']

train = train.merge(passthrough, on='account_id', how='left')

mule_pt  = train[train['is_mule']==1]['passthrough_rate'].median()
legit_pt = train[train['is_mule']==0]['passthrough_rate'].median()
print(f"Mule median pass-through rate:  {mule_pt:.2%}")
print(f"Legit median pass-through rate: {legit_pt:.2%}")
print("(Higher = money comes in and goes out same day)")

# ============================================================
# ANALYSIS 5: ROUND AMOUNT PATTERN
# Mules use suspiciously round numbers: 10000, 50000, etc.
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS 5: ROUND AMOUNT PATTERNS")
print("=" * 60)

transactions['is_round'] = transactions['amount'] % 1000 == 0
round_pct = transactions.groupby('account_id')['is_round'].mean().reset_index()
round_pct.columns = ['account_id', 'round_amount_pct']

train = train.merge(round_pct, on='account_id', how='left')

mule_rnd  = train[train['is_mule']==1]['round_amount_pct'].median()
legit_rnd = train[train['is_mule']==0]['round_amount_pct'].median()
print(f"Mule median round-amount %:  {mule_rnd:.2%}")
print(f"Legit median round-amount %: {legit_rnd:.2%}")

# ============================================================
# ANALYSIS 6: DORMANT ACCOUNT DETECTION
# Long gap of inactivity, then sudden burst of transactions
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS 6: DORMANT ACCOUNT REACTIVATION")
print("=" * 60)

# Find the largest gap between consecutive transactions per account
txn_sorted2 = transactions.sort_values(['account_id', 'transaction_timestamp'])
txn_sorted2['prev_ts'] = txn_sorted2.groupby('account_id')['transaction_timestamp'].shift(1)
txn_sorted2['gap_days'] = (txn_sorted2['transaction_timestamp'] - txn_sorted2['prev_ts']).dt.days

dormancy = txn_sorted2.groupby('account_id')['gap_days'].max().reset_index()
dormancy.columns = ['account_id', 'max_gap_days']

train = train.merge(dormancy, on='account_id', how='left')

mule_gap  = train[train['is_mule']==1]['max_gap_days'].median()
legit_gap = train[train['is_mule']==0]['max_gap_days'].median()
print(f"Mule median max gap between txns:  {mule_gap:.0f} days")
print(f"Legit median max gap between txns: {legit_gap:.0f} days")

# ============================================================
# ANALYSIS 7: POST MOBILE UPDATE SPIKE
# Transactions surge after mobile number change = account takeover
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS 7: POST MOBILE UPDATE SPIKE")
print("=" * 60)

train['last_mobile_update_date'] = pd.to_datetime(train['last_mobile_update_date'], errors='coerce')
has_mobile_update = train['last_mobile_update_date'].notna()

print(f"Accounts with mobile update — Mule: {train[train['is_mule']==1][has_mobile_update].shape[0]} / {train[train['is_mule']==1].shape[0]}")
print(f"Accounts with mobile update — Legit: {train[train['is_mule']==0][has_mobile_update].shape[0]} / {train[train['is_mule']==0].shape[0]}")

mule_mob_pct  = has_mobile_update[train['is_mule']==1].mean()
legit_mob_pct = has_mobile_update[train['is_mule']==0].mean()
print(f"Mule % with mobile update:  {mule_mob_pct:.2%}")
print(f"Legit % with mobile update: {legit_mob_pct:.2%}")

# ============================================================
# ANALYSIS 8: CHANNEL USAGE BREAKDOWN
# Which transaction channels do mules prefer?
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS 8: CHANNEL USAGE")
print("=" * 60)

mule_accounts  = train[train['is_mule']==1]['account_id'].tolist()
legit_accounts = train[train['is_mule']==0]['account_id'].tolist()

mule_txns  = transactions[transactions['account_id'].isin(mule_accounts)]
legit_txns = transactions[transactions['account_id'].isin(legit_accounts)]

mule_channels  = mule_txns['channel'].value_counts(normalize=True).head(8)
legit_channels = legit_txns['channel'].value_counts(normalize=True).head(8)

channel_compare = pd.DataFrame({
    'Mule %':  mule_channels,
    'Legit %': legit_channels
}).fillna(0) * 100

print(channel_compare.round(2).to_string())

# ============================================================
# ANALYSIS 9: PRODUCT FAMILY BREAKDOWN
# S=Savings, K=K-family, O=Overdraft
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS 9: PRODUCT FAMILY")
print("=" * 60)

prod_cross = pd.crosstab(train['product_family'], train['is_mule'], normalize='columns') * 100
print(prod_cross.round(2))

# ============================================================
# PLOT 1: KEY FEATURE COMPARISON (save as chart1.png)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Mule vs Legitimate Account Comparison', fontsize=16, fontweight='bold')

plot_data = [
    ('avg_txn_amount',        'Avg Transaction Amount (₹)', axes[0,0]),
    ('unique_counterparties', 'Unique Counterparties',       axes[0,1]),
    ('total_txn_count',       'Total Transaction Count',     axes[0,2]),
    ('passthrough_rate',      'Pass-Through Rate',           axes[1,0]),
    ('round_amount_pct',      'Round Amount %',              axes[1,1]),
    ('account_age_days',      'Account Age (days)',          axes[1,2]),
]

for col, title, ax in plot_data:
    mule_vals  = train[train['is_mule']==1][col].dropna()
    legit_vals = train[train['is_mule']==0][col].dropna()

    # Use 95th percentile to cut extreme outliers for readability
    cap = np.percentile(legit_vals, 95)
    mule_vals  = mule_vals.clip(upper=cap)
    legit_vals = legit_vals.clip(upper=cap)

    ax.hist(legit_vals, bins=40, alpha=0.6, color='steelblue', label='Legit', density=True)
    ax.hist(mule_vals,  bins=40, alpha=0.7, color='crimson',   label='Mule',  density=True)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Density')
    ax.legend()
    ax.axvline(mule_vals.median(),  color='crimson',   linestyle='--', linewidth=1.5)
    ax.axvline(legit_vals.median(), color='steelblue', linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.savefig('chart1_feature_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: chart1_feature_comparison.png")

# ============================================================
# PLOT 2: FROZEN ACCOUNT + PRODUCT FAMILY (save as chart2.png)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Account Status & Product Family', fontsize=14, fontweight='bold')

# Frozen accounts
status_data = pd.crosstab(train['account_status'], train['is_mule'], normalize='columns') * 100
status_data.plot(kind='bar', ax=axes[0], color=['steelblue', 'crimson'], rot=0)
axes[0].set_title('Account Status Distribution')
axes[0].set_ylabel('% of accounts')
axes[0].legend(['Legit', 'Mule'])

# Product family
prod_data = pd.crosstab(train['product_family'], train['is_mule'], normalize='columns') * 100
prod_data.plot(kind='bar', ax=axes[1], color=['steelblue', 'crimson'], rot=0)
axes[1].set_title('Product Family Distribution')
axes[1].set_ylabel('% of accounts')
axes[1].legend(['Legit', 'Mule'])

plt.tight_layout()
plt.savefig('chart2_status_product.png', dpi=150, bbox_inches='tight')
print("✅ Saved: chart2_status_product.png")

# ============================================================
# PLOT 3: CHANNEL USAGE (save as chart3.png)
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
channel_compare.plot(kind='bar', ax=ax, color=['steelblue', 'crimson'], rot=45)
ax.set_title('Transaction Channel Usage: Mule vs Legit (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('% of transactions')
ax.legend(['Mule %', 'Legit %'])
plt.tight_layout()
plt.savefig('chart3_channels.png', dpi=150, bbox_inches='tight')
print("✅ Saved: chart3_channels.png")

# ============================================================
# FINAL SUMMARY — print everything for your report
# ============================================================

print("\n" + "=" * 60)
print("FULL SUMMARY FOR YOUR REPORT")
print("=" * 60)

summary = {
    "Mule Rate": f"{labels['is_mule'].mean():.2%}",
    "Frozen Account Rate (Mule)":  f"{train[train['is_mule']==1]['account_status'].eq('frozen').mean():.2%}",
    "Frozen Account Rate (Legit)": f"{train[train['is_mule']==0]['account_status'].eq('frozen').mean():.2%}",
    "Avg Txn Amount - Mule":  f"₹{train[train['is_mule']==1]['avg_txn_amount'].median():,.0f}",
    "Avg Txn Amount - Legit": f"₹{train[train['is_mule']==0]['avg_txn_amount'].median():,.0f}",
    "Unique Counterparties - Mule":  f"{train[train['is_mule']==1]['unique_counterparties'].median():.0f}",
    "Unique Counterparties - Legit": f"{train[train['is_mule']==0]['unique_counterparties'].median():.0f}",
    "Pass-Through Rate - Mule":  f"{train[train['is_mule']==1]['passthrough_rate'].median():.2%}",
    "Pass-Through Rate - Legit": f"{train[train['is_mule']==0]['passthrough_rate'].median():.2%}",
    "Round Amount % - Mule":  f"{train[train['is_mule']==1]['round_amount_pct'].median():.2%}",
    "Round Amount % - Legit": f"{train[train['is_mule']==0]['round_amount_pct'].median():.2%}",
    "Max Dormancy Gap - Mule":  f"{train[train['is_mule']==1]['max_gap_days'].median():.0f} days",
    "Max Dormancy Gap - Legit": f"{train[train['is_mule']==0]['max_gap_days'].median():.0f} days",
}

for k, v in summary.items():
    print(f"  {k:40s}: {v}")

print("\n✅ Day 2 complete! Charts saved. Paste all the output numbers here for Day 3.")