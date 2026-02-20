# Mule Account Detection — Phase 1: EDA

**Financial Crime Detection · IIT Delhi Hackathon**

Exploratory data analysis on a 20% sample of a real-world banking dataset to identify mule accounts used in money laundering.

---

## Dataset Overview

| Table | Rows | Description |
|:--|--:|:--|
| `customers.csv` | 39,988 | Demographics, KYC flags, banking registrations |
| `accounts.csv` | 40,038 | Account attributes, balance metrics, status |
| `transactions` (×6 parts) | 7,424,845 | Every transaction — channel, amount, counterparty |
| `customer_account_linkage.csv` | 40,038 | Bridge: maps customers → accounts |
| `product_details.csv` | 39,988 | Product holdings: loans, credit cards, overdraft |
| `train_labels.csv` | 24,023 | Ground truth: `is_mule` flag, flag date, alert reason |
| `test_accounts.csv` | 16,015 | Accounts to predict on in Phase 2 |

**Class imbalance:** 263 mule accounts (1.09%) vs 23,760 legitimate (98.91%) — ratio 90:1.

---

## Key Findings

| Finding | Signal |
|:--|:--|
| Frozen account rate (40% mule vs 2% legit) | 🔴 Very Strong |
| Pass-through rate — money in & out same day (7.5% vs 0%) | 🔴 Very Strong |
| Unique counterparties (30 vs 10 median) | 🔴 Strong |
| Avg transaction amount (₹14,845 vs ₹7,343) | 🔴 Strong |
| ATM withdrawals present (1.69% vs 0%) — novel finding | 🔵 Moderate |
| 7 of 12 known mule patterns confirmed in data | ✅ |

---

## Repo Structure

```
├── eda1.py                   # Initial data exploration
├── eda2.py                   # Feature analysis & pattern testing
├── eda_report_clean.ipynb    # Final EDA report notebook (pure markdown + charts)
├── eda_report_clean.html     # Styled HTML export (open in browser → Print to PDF)
├── export_html.py            # Script to re-export notebook to styled HTML
└── data/                     # Raw data files (not tracked by git)
```

---

## How to Run

```bash
# 1. Install dependencies
pip install matplotlib nbformat nbconvert markdown

# 2. Open the EDA notebook
jupyter notebook eda_report_clean.ipynb

# 3. Re-export styled HTML (after any edits)
python3 export_html.py eda_report_clean.ipynb
# Then open eda_report_clean.html in Chrome → ⌘P → Save as PDF
```

---

## Phase 2 Plan

- Feature engineering (20+ features derived from transaction behaviour)
- XGBoost / LightGBM classifier
- Handle 90:1 imbalance via SMOTE + class weights
- Evaluation metric: AUC-ROC / PR-AUC
