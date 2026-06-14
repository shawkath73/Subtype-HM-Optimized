"""
evaluate_brca.py
Computes clustering efficiency metrics (ACC %, NMI, ARI) by comparing
the model's predicted subtypes (results/BRCA.dcc) against the ground-truth
PAM50 labels stored in the BRCA mRNA CSV (Label column).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score,
)
from scipy.optimize import linear_sum_assignment
import os

# ── 1. Load ground-truth labels from the mRNA CSV ─────────────────────────
CACHE_PATH = os.path.join('data', 'brca_cache', 'brca_mrna.csv')
print(f"Loading ground-truth labels from: {CACHE_PATH}")
df_feat = pd.read_csv(CACHE_PATH, header=0, index_col=None)
label_col = df_feat.columns[-1]          # last column = 'Label'
true_labels = df_feat[label_col].values.astype(int)

# The model only uses the first 875 samples (min across 3 views)
n_samples = 875
true_labels = true_labels[:n_samples]

# ── 2. Load predicted cluster assignments from results/BRCA.dcc ───────────
DCC_PATH = os.path.join('results', 'BRCA.dcc')
print(f"Loading predictions from: {DCC_PATH}")
df_pred = pd.read_csv(DCC_PATH, sep='\t')
pred_labels = df_pred['dcc'].values.astype(int)[:n_samples]

print(f"\nSamples evaluated : {n_samples}")
print(f"Unique true classes: {sorted(set(true_labels))}")
print(f"Unique pred classes: {sorted(set(pred_labels))}")

# ── 3. Clustering ACC via Hungarian matching ───────────────────────────────
def clustering_accuracy(y_true, y_pred):
    """Best-match accuracy using the Hungarian algorithm."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    cost = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cost[t][p] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)   # maximise hits
    total_correct = cost[row_ind, col_ind].sum()
    return total_correct / len(y_true)

acc  = clustering_accuracy(true_labels, pred_labels)
nmi  = normalized_mutual_info_score(true_labels, pred_labels)
ari  = adjusted_rand_score(true_labels, pred_labels)

# ── 4. Load KMeans baseline if available ──────────────────────────────────
KMEANS_DCC_PATH = os.path.join('results', 'KMeans_baseline.dcc')
has_baseline = False
if os.path.exists(KMEANS_DCC_PATH):
    print(f"Loading KMeans baseline predictions from: {KMEANS_DCC_PATH}")
    df_base = pd.read_csv(KMEANS_DCC_PATH, sep='\t')
    base_labels = df_base['dcc'].values.astype(int)[:n_samples]
    acc_base = clustering_accuracy(true_labels, base_labels)
    nmi_base = normalized_mutual_info_score(true_labels, base_labels)
    ari_base = adjusted_rand_score(true_labels, base_labels)
    has_baseline = True
else:
    print(f"\n[!] KMeans baseline not found at {KMEANS_DCC_PATH}.")
    print("    Run 'python run_kmeans_baseline.py' first to generate it.")

# ── 5. Print results ───────────────────────────────────────────────────────
if has_baseline:
    print("\n" + "="*60)
    print("              BRCA Subtyping Efficiency Metrics")
    print("="*60)
    print(f"{'Metric':<28} | {'Subtype-HM':<12} | {'KMeans Baseline':<15}")
    print("-" * 60)
    print(f"{'Clustering Accuracy (ACC)':<28} | {acc*100:6.2f}%       | {acc_base*100:6.2f}%")
    print(f"{'Normalized Mutual Info (NMI)':<28} | {nmi:8.4f}     | {nmi_base:8.4f}")
    print(f"{'Adjusted Rand Index (ARI)':<28} | {ari:8.4f}     | {ari_base:8.4f}")
    print("="*60)
else:
    print("\n" + "="*45)
    print("        BRCA Subtyping Efficiency Metrics")
    print("="*45)
    print(f"  Clustering Accuracy (ACC) : {acc*100:.2f} %")
    print(f"  Normalized Mutual Info    : {nmi:.4f}  (0–1, higher=better)")
    print(f"  Adjusted Rand Index (ARI) : {ari:.4f}  (-1–1, higher=better)")
    print("="*45)

print("\nNote: ACC uses Hungarian algorithm (best label permutation match).")
print("Ground-truth labels = PAM50 subtypes from the CSV Label column.")
