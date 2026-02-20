import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATA = "data/"  # all your CSVs should be inside the data/ folder

# ============================================================
# STEP 2: LOAD ALL DATA
# ============================================================

customers    = pd.read_csv(f"{DATA}customers.csv")
accounts     = pd.read_csv(f"{DATA}accounts.csv")
linkage      = pd.read_csv(f"{DATA}customer_account_linkage.csv")
products     = pd.read_csv(f"{DATA}product_details.csv")
labels       = pd.read_csv(f"{DATA}train_labels.csv")
test         = pd.read_csv(f"{DATA}test_accounts.csv")

transactions = pd.concat(
    [pd.read_csv(f"{DATA}transactions_part_{i}.csv") for i in range(6)],
    ignore_index=True
)

print("Data loaded!")
print(f"  Customers:    {customers.shape}")
print(f"  Accounts:     {accounts.shape}")
print(f"  Transactions: {transactions.shape}")
print(f"  Labels:       {labels.shape}")

# ============================================================
# STEP 3: UNDERSTAND THE TARGET (mule rate)
# ============================================================

mule_count = labels['is_mule'].sum()
total      = len(labels)
print(f"\nMule accounts: {mule_count} / {total} = {mule_count/total:.2%}")

# ============================================================
# STEP 4: BUILD YOUR MASTER TRAINING TABLE
# ============================================================

train = labels.copy()
train = train.merge(accounts,  on="account_id",  how="left")
train = train.merge(linkage,   on="account_id",  how="left")
train = train.merge(customers, on="customer_id", how="left")
train = train.merge(products,  on="customer_id", how="left")

print(f"\nMaster training table shape: {train.shape}")
print(train.head(2))

# ============================================================
# STEP 5: CHECK MISSING VALUES
# ============================================================

missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\nMissing values:")
print(missing)

# ============================================================
# STEP 6: COMPARE MULE vs LEGIT — Account attributes
# ============================================================

flag_cols = ['nomination_flag', 'kyc_compliant', 'cheque_allowed',
             'cheque_availed', 'rural_branch', 'pan_available',
             'aadhaar_available', 'mobile_banking_flag', 'internet_banking_flag',
             'atm_card_flag', 'credit_card_flag']

print("\n--- FLAG COMPARISONS: Mule vs Legit ---")
for col in flag_cols:
    if col in train.columns:
        mule_rate  = train[train['is_mule']==1][col].eq('Y').mean()
        legit_rate = train[train['is_mule']==0][col].eq('Y').mean()
        print(f"{col:35s}  Mule: {mule_rate:.1%}   Legit: {legit_rate:.1%}")

# ============================================================
# STEP 7: COMPARE BALANCE DISTRIBUTIONS
# ============================================================

print("\n--- BALANCE COMPARISONS: Mule vs Legit ---")
for col in ['avg_balance', 'monthly_avg_balance', 'daily_avg_balance']:
    mule_med  = train[train['is_mule']==1][col].median()
    legit_med = train[train['is_mule']==0][col].median()
    print(f"{col:30s}  Mule median: {mule_med:>10.1f}   Legit median: {legit_med:>10.1f}")

# ============================================================
# STEP 8: ACCOUNT STATUS
# ============================================================

print("\n--- ACCOUNT STATUS ---")
print(pd.crosstab(train['account_status'], train['is_mule'], normalize='columns'))

# ============================================================
# STEP 9: TRANSACTION FEATURES
# ============================================================

transactions['transaction_timestamp'] = pd.to_datetime(transactions['transaction_timestamp'])

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

txn_features['credit_debit_ratio'] = (
    txn_features['credit_count'] / (txn_features['debit_count'] + 1)
)

train = train.merge(txn_features, on='account_id', how='left')

print("\n--- TRANSACTION BEHAVIOR: Mule vs Legit ---")
for col in ['total_txn_count', 'avg_txn_amount', 'unique_counterparties', 'credit_debit_ratio']:
    mule_med  = train[train['is_mule']==1][col].median()
    legit_med = train[train['is_mule']==0][col].median()
    print(f"{col:30s}  Mule: {mule_med:>10.2f}   Legit: {legit_med:>10.2f}")

# ============================================================
# STEP 10: DETECT STRUCTURING (transactions near 50,000)
# ============================================================

transactions['near_50k'] = transactions['amount'].between(45000, 50000)
structuring = transactions.groupby('account_id')['near_50k'].sum().reset_index()
structuring.columns = ['account_id', 'near_50k_count']
train = train.merge(structuring, on='account_id', how='left')

print("\n--- STRUCTURING (txns near 50K): Mule vs Legit ---")
print(f"Mule median near-50k txns:  {train[train['is_mule']==1]['near_50k_count'].median()}")
print(f"Legit median near-50k txns: {train[train['is_mule']==0]['near_50k_count'].median()}")

print("\n✅ Day 1 complete! You now have a master table with transaction features.")
print(f"Final training table shape: {train.shape}")