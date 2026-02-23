"""
Script to add/replace matplotlib chart cells in eda_report_clean.ipynb
for sections 4a, 4b, 4c, and 6a per user specifications.
"""
import json, copy

NB_PATH = "eda_report_clean.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Helper: create a code cell with source hidden + hide-input tag ──
def make_code_cell(source_lines, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {
            "jupyter": {"source_hidden": True},
            "tags": ["hide-input"]
        },
        "outputs": [],
        "source": source_lines
    }

# ── Chart 1 — Section 4a: Box plot of burst_ratio ──
chart1_src = [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "MULE = '#dc2626'; LEGIT = '#059669'\n",
    "\n",
    "mule_br = train[train['is_mule'] == 1]['burst_ratio'].dropna()\n",
    "legit_br = train[train['is_mule'] == 0]['burst_ratio'].dropna()\n",
    "\n",
    "# Cap y-axis at 95th percentile to avoid outlier distortion\n",
    "cap = np.percentile(list(mule_br) + list(legit_br), 95)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(7, 5))\n",
    "bp = ax.boxplot(\n",
    "    [mule_br.values, legit_br.values],\n",
    "    labels=['Mule', 'Legitimate'],\n",
    "    patch_artist=True, widths=0.45,\n",
    "    showfliers=False\n",
    ")\n",
    "bp['boxes'][0].set_facecolor(MULE); bp['boxes'][0].set_alpha(0.6)\n",
    "bp['boxes'][1].set_facecolor(LEGIT); bp['boxes'][1].set_alpha(0.6)\n",
    "for median_line in bp['medians']:\n",
    "    median_line.set_color('black'); median_line.set_linewidth(2)\n",
    "\n",
    "ax.set_ylim(top=cap)\n",
    "ax.set_ylabel('Burst Ratio')\n",
    "ax.set_title('Burst Ratio Distribution: Mule vs Legitimate', fontweight='bold')\n",
    "ax.grid(axis='y', alpha=0.3)\n",
    "ax.spines[['top', 'right']].set_visible(False)\n",
    "\n",
    "# Annotate medians\n",
    "med_m = float(np.median(mule_br))\n",
    "med_l = float(np.median(legit_br))\n",
    "ax.text(1, med_m + cap * 0.03, f'Median: {med_m:.2f}',\n",
    "        ha='center', fontsize=9, fontweight='bold', color=MULE)\n",
    "ax.text(2, med_l + cap * 0.03, f'Median: {med_l:.2f}',\n",
    "        ha='center', fontsize=9, fontweight='bold', color=LEGIT)\n",
    "\n",
    "plt.tight_layout(); plt.show()\n"
]

# ── Chart 2 — Section 4b: Side-by-side bar chart (two panels) ──
chart2_src = [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "MULE = '#dc2626'; LEGIT = '#059669'\n",
    "\n",
    "mule_mask = train['is_mule'] == 1\n",
    "legit_mask = train['is_mule'] == 0\n",
    "\n",
    "med_entropy_m = float(train.loc[mule_mask, 'counterparty_entropy'].median())\n",
    "med_entropy_l = float(train.loc[legit_mask, 'counterparty_entropy'].median())\n",
    "med_top3_m = float(train.loc[mule_mask, 'pct_volume_top3'].median())\n",
    "med_top3_l = float(train.loc[legit_mask, 'pct_volume_top3'].median())\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))\n",
    "fig.suptitle('Network Structure: Mule vs Legitimate',\n",
    "             fontsize=12, fontweight='bold', y=1.02)\n",
    "\n",
    "# Left panel — counterparty_entropy\n",
    "bars1 = ax1.bar(['Mule', 'Legit'], [med_entropy_m, med_entropy_l],\n",
    "               color=[MULE, LEGIT], width=0.5, edgecolor='white')\n",
    "for b in bars1:\n",
    "    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,\n",
    "             f'{b.get_height():.3f}', ha='center', fontsize=10,\n",
    "             fontweight='bold', color='#333')\n",
    "ax1.set_ylabel('Median Counterparty Entropy')\n",
    "ax1.set_title('Counterparty Entropy', fontweight='bold')\n",
    "ax1.grid(axis='y', alpha=0.3)\n",
    "ax1.spines[['top', 'right']].set_visible(False)\n",
    "\n",
    "# Right panel — pct_volume_top3\n",
    "bars2 = ax2.bar(['Mule', 'Legit'], [med_top3_m, med_top3_l],\n",
    "               color=[MULE, LEGIT], width=0.5, edgecolor='white')\n",
    "for b in bars2:\n",
    "    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,\n",
    "             f'{b.get_height():.3f}', ha='center', fontsize=10,\n",
    "             fontweight='bold', color='#333')\n",
    "ax2.set_ylabel('Median % Volume in Top 3')\n",
    "ax2.set_title('Volume Concentration (Top 3)', fontweight='bold')\n",
    "ax2.grid(axis='y', alpha=0.3)\n",
    "ax2.spines[['top', 'right']].set_visible(False)\n",
    "\n",
    "plt.tight_layout(); plt.show()\n"
]

# ── Chart 3 — Section 4c: Horizontal bar of branch relative risk ──
chart3_src = [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "bf = branch_stats[branch_stats['total_accounts'] >= 30].copy()\n",
    "bf['mule_rate'] = bf['mule_count'] / bf['total_accounts']\n",
    "bf['relative_risk'] = bf['mule_rate'] / global_mule_rate\n",
    "\n",
    "# Top 20 branches by account count for readability\n",
    "bf = bf.nlargest(20, 'total_accounts').sort_values('relative_risk')\n",
    "\n",
    "colors = ['#dc2626' if rr > 2 else '#94a3b8' for rr in bf['relative_risk']]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(9, 6))\n",
    "ax.barh(bf['branch_code'].astype(str), bf['relative_risk'],\n",
    "        color=colors, height=0.6, edgecolor='white')\n",
    "ax.axvline(x=1.0, color='#333', ls='--', lw=1.2, label='Global baseline (RR = 1)')\n",
    "ax.set_xlabel('Relative Risk (branch mule rate / global mule rate)')\n",
    "ax.set_ylabel('Branch Code')\n",
    "ax.set_title('Branch Relative Risk \\u2014 Mule Concentration', fontweight='bold')\n",
    "ax.legend(fontsize=9, loc='lower right')\n",
    "ax.grid(axis='x', alpha=0.3)\n",
    "ax.spines[['top', 'right']].set_visible(False)\n",
    "plt.tight_layout(); plt.show()\n"
]

# ── Chart 4 — Section 6a: Cliff's delta horizontal bar chart ──
chart4_src = [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from scipy.stats import mannwhitneyu\n",
    "\n",
    "features_to_rank = ['burst_ratio', 'longest_inactivity_gap', 'active_span_days',\n",
    "    'counterparty_entropy', 'gini_volume', 'pct_volume_top3', 'pct_txn_day_1_5']\n",
    "\n",
    "def cliffs_delta_calc(x, y):\n",
    "    nx, ny = len(x), len(y)\n",
    "    gt = sum(1 for xi in x for yi in y if xi > yi)\n",
    "    lt = sum(1 for xi in x for yi in y if xi < yi)\n",
    "    return (gt - lt) / (nx * ny)\n",
    "\n",
    "rows = []\n",
    "for feat in features_to_rank:\n",
    "    mv = train[train['is_mule']==1][feat].dropna().values\n",
    "    lv = train[train['is_mule']==0][feat].dropna().values\n",
    "    if len(mv) == 0 or len(lv) == 0: continue\n",
    "    d = cliffs_delta_calc(mv, lv)\n",
    "    rows.append({'Feature': feat, 'Cliff_Delta': round(d, 4),\n",
    "        'Abs_Delta': round(abs(d), 4)})\n",
    "\n",
    "rank_df = pd.DataFrame(rows).sort_values('Abs_Delta', ascending=True)\n",
    "\n",
    "# Color: green if |delta| > 0.1 (signal), grey for null\n",
    "colors = ['#059669' if ad > 0.1 else '#94a3b8' for ad in rank_df['Abs_Delta']]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(9, 5))\n",
    "ax.barh(rank_df['Feature'], rank_df['Cliff_Delta'],\n",
    "        color=colors, height=0.55, edgecolor='white')\n",
    "ax.axvline(x=0, color='#333', ls='--', lw=1.2, label='\\u03b4 = 0')\n",
    "ax.set_xlabel(\"Cliff's Delta\")\n",
    "ax.set_title(\"Feature Separation by Cliff's Delta (Effect Size)\", fontweight='bold')\n",
    "ax.legend(fontsize=9, loc='lower right')\n",
    "ax.grid(axis='x', alpha=0.3)\n",
    "ax.spines[['top', 'right']].set_visible(False)\n",
    "plt.tight_layout(); plt.show()\n"
]

# ── Locate cells by section markers and replace/insert ──

def find_cell_index(cells, marker_text):
    """Find the index of a markdown cell whose source contains marker_text."""
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown":
            src = "".join(c.get("source", []))
            if marker_text in src:
                return i
    return None

def replace_or_insert_code_after_markdown(cells, marker, new_code_src, cell_id):
    """
    Find markdown cell with `marker`, then:
    - if next cell is code, replace its source & clear outputs
    - otherwise insert a new code cell after it
    """
    idx = find_cell_index(cells, marker)
    if idx is None:
        print(f"WARNING: marker '{marker}' not found")
        return

    # Check if there's an existing text cell after the markdown header
    # that acts as body text — we want to insert AFTER ALL consecutive
    # markdown cells in that section, but BEFORE the next code cell.
    insert_pos = idx + 1
    while insert_pos < len(cells) and cells[insert_pos]["cell_type"] == "markdown":
        insert_pos += 1

    new_cell = make_code_cell(new_code_src, cell_id)

    if insert_pos < len(cells) and cells[insert_pos]["cell_type"] == "code":
        # Replace existing code cell
        cells[insert_pos] = new_cell
        print(f"Replaced code cell after '{marker}' at index {insert_pos}")
    else:
        # Insert new code cell
        cells.insert(insert_pos, new_cell)
        print(f"Inserted code cell after '{marker}' at index {insert_pos}")


# Chart 1 → after section 4a text
replace_or_insert_code_after_markdown(
    cells, "## 4a. Temporal Burst Behaviour", chart1_src, "chart_4a_boxplot")

# Chart 2 → after section 4b text
replace_or_insert_code_after_markdown(
    cells, "## 4b. Network Structure Analysis", chart2_src, "chart_4b_bars")

# Chart 3 → after section 4c text
replace_or_insert_code_after_markdown(
    cells, "## 4c. Branch-Level Risk Assessment", chart3_src, "chart_4c_branch")

# Chart 4 → after section 6a text
replace_or_insert_code_after_markdown(
    cells, "## 6a. Top Behavioural Predictors by Empirical Separation", chart4_src, "chart_6a_cliff")

# ── Write back ──
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✅ Notebook updated successfully. Run all cells to generate charts.")
