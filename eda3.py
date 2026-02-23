# ===============================
# EDA3: Advanced Behavioural Analysis
# ===============================

# ===============================
# IMPORTS
# ===============================
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp

# ===============================
# LOAD DATA
# ===============================

accounts = pd.read_csv("data/accounts.csv")
customers = pd.read_csv("data/customers.csv")
linkage = pd.read_csv("data/customer_account_linkage.csv")
products = pd.read_csv("data/product_details.csv")
labels = pd.read_csv("data/train_labels.csv")

# Load transactions
transactions = pd.concat(
    [pd.read_csv(f"data/transactions_part_{i}.csv") for i in range(6)],
    ignore_index=True
)

# 🔥 Convert timestamp AFTER concat
transactions["transaction_timestamp"] = pd.to_datetime(
    transactions["transaction_timestamp"],
    errors="coerce"
)

# ===============================
# BUILD MASTER TRAIN TABLE
# ===============================

train = labels.merge(accounts, on="account_id", how="left")
train = train.merge(linkage, on="account_id", how="left")
train = train.merge(customers, on="customer_id", how="left")
train = train.merge(products, on="customer_id", how="left")

print("Data loaded successfully.")
print("Transactions shape:", transactions.shape)
print("Train shape:", train.shape)

# 1️⃣ TEMPORAL WINDOW FEATURES
# =====================================================

transactions["year_month"] = transactions["transaction_timestamp"].dt.to_period("M")

monthly_txn = (
    transactions
    .groupby(["account_id", "year_month"])
    .size()
    .reset_index(name="monthly_txn_count")
)

burst_stats = (
    monthly_txn
    .groupby("account_id")["monthly_txn_count"]
    .agg(["max", "median"])
    .reset_index()
)

burst_stats["burst_ratio"] = burst_stats["max"] / burst_stats["median"].replace(0, np.nan)

activity_span = (
    transactions
    .groupby("account_id")["transaction_timestamp"]
    .agg(["min", "max"])
    .reset_index()
)

activity_span["active_span_days"] = (
    activity_span["max"] - activity_span["min"]
).dt.days

transactions["prev_txn"] = transactions.groupby("account_id")["transaction_timestamp"].shift()
transactions["gap_days"] = (
    transactions["transaction_timestamp"] - transactions["prev_txn"]
).dt.days

longest_gap = (
    transactions
    .groupby("account_id")["gap_days"]
    .max()
    .reset_index()
    .rename(columns={"gap_days": "longest_inactivity_gap"})
)

temporal_features = (
    burst_stats
    .merge(activity_span[["account_id", "active_span_days"]], on="account_id")
    .merge(longest_gap, on="account_id")
)

# =====================================================
# 2️⃣ SALARY WINDOW FEATURES (1st–5th)
# =====================================================

transactions["day"] = transactions["transaction_timestamp"].dt.day
transactions["is_salary_window"] = transactions["day"].between(1, 5)

salary_stats = (
    transactions
    .groupby("account_id")
    .agg(
        pct_txn_day_1_5=("is_salary_window", "mean"),
        pct_credit_day_1_5=("txn_type", lambda x: ((x=="C") & transactions.loc[x.index,"is_salary_window"]).mean()),
        pct_debit_day_1_5=("txn_type", lambda x: ((x=="D") & transactions.loc[x.index,"is_salary_window"]).mean())
    )
    .reset_index()
)

# =====================================================
# 3️⃣ NETWORK DEPTH FEATURES
# =====================================================

def entropy(series):
    counts = series.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts + 1e-9))

entropy_df = (
    transactions
    .groupby("account_id")["counterparty_id"]
    .apply(entropy)
    .reset_index(name="counterparty_entropy")
)

def gini(array):
    array = np.array(array)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-9
    array = np.sort(array)
    n = len(array)
    return (np.sum((2*np.arange(1, n+1)-n-1) * array)) / (n * np.sum(array))

gini_df = (
    transactions
    .groupby(["account_id", "counterparty_id"])["amount"]
    .sum()
    .groupby("account_id")
    .apply(lambda x: gini(x.values))
    .reset_index(name="gini_volume")
)

top3_df = (
    transactions
    .groupby(["account_id", "counterparty_id"])["amount"]
    .sum()
    .reset_index()
)

top3_pct = (
    top3_df
    .sort_values(["account_id", "amount"], ascending=[True, False])
    .groupby("account_id")
    .apply(lambda x: x.head(3)["amount"].sum() / x["amount"].sum())
    .reset_index(name="pct_volume_top3")
)

network_features = (
    entropy_df
    .merge(gini_df, on="account_id")
    .merge(top3_pct, on="account_id")
)

# =====================================================
# 4️⃣ MERGE ALL FEATURES INTO TRAIN
# =====================================================

train = train.merge(temporal_features, on="account_id", how="left")
train = train.merge(salary_stats, on="account_id", how="left")
train = train.merge(network_features, on="account_id", how="left")

# =====================================================
# 5️⃣ BRANCH COLLUSION Z-SCORE
# =====================================================

global_mule_rate = train["is_mule"].mean()

branch_stats = (
    train
    .groupby("branch_code")
    .agg(
        total_accounts=("account_id", "count"),
        mule_count=("is_mule", "sum")
    )
    .reset_index()
)

branch_stats["expected_mules"] = branch_stats["total_accounts"] * global_mule_rate

branch_stats["z_score"] = (
    (branch_stats["mule_count"] - branch_stats["expected_mules"])
    / np.sqrt(branch_stats["total_accounts"] * global_mule_rate * (1 - global_mule_rate))
)

abnormal_branches = branch_stats[branch_stats["z_score"].abs() > 2]

print("\nAbnormal branches (|z| > 2):")
print(abnormal_branches)

# =====================================================
# 6️⃣ STATISTICAL TESTING UTILITIES
# =====================================================

def compare_feature(feature):
    mule = train[train["is_mule"] == 1][feature].dropna()
    legit = train[train["is_mule"] == 0][feature].dropna()

    if len(mule) == 0 or len(legit) == 0:
        print(f"\nFeature: {feature} — insufficient data")
        return

    u_stat, p_mw = mannwhitneyu(mule, legit, alternative="two-sided")
    ks_stat, p_ks = ks_2samp(mule, legit)

    median_diff = mule.median() - legit.median()

    print(f"\nFeature: {feature}")
    print(f"Mule median: {mule.median():.4f}")
    print(f"Legit median: {legit.median():.4f}")
    print(f"Median difference: {median_diff:.4f}")
    print(f"Mann-Whitney p-value: {p_mw:.6f}")
    print(f"KS p-value: {p_ks:.6f}")

def cliffs_delta(x, y):
    nx = len(x)
    ny = len(y)
    greater = sum(xi > yi for xi in x for yi in y)
    less = sum(xi < yi for xi in x for yi in y)
    return (greater - less) / (nx * ny)

def effect_size(feature):
    mule = train[train["is_mule"] == 1][feature].dropna().values
    legit = train[train["is_mule"] == 0][feature].dropna().values
    if len(mule) == 0 or len(legit) == 0:
        print(f"{feature} — insufficient data")
        return
    delta = cliffs_delta(mule, legit)
    print(f"{feature} Cliff's Delta: {delta:.4f}")

# =====================================================
# 7️⃣ RUN COMPARISONS FOR NEW FEATURES
# =====================================================

features_to_test = [
    "burst_ratio",
    "longest_inactivity_gap",
    "active_span_days",
    "counterparty_entropy",
    "gini_volume",
    "pct_volume_top3",
    "pct_txn_day_1_5"
]

for f in features_to_test:
    compare_feature(f)
    effect_size(f)

print("\nEDA3 feature generation complete.")