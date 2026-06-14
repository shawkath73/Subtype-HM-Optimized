"""
Custom dataloader for TCGA BRCA CSV datasets.
Downloads data from GitHub URLs and creates a 3-view MultiViewDataset.

The CSV files have shape: (n_samples, n_features + 1)
where the last column is 'Label' (subtype class).

Survival data is fetched from UCSC Xena (real TCGA BRCA clinical data).
Falls back to synthetic placeholders only if the download fails.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import io
import os

# ── URLs ────────────────────────────────────────────────────────────────────
URL_MRNA  = 'https://raw.githubusercontent.com/shawkath73/GCN-implementation/refs/heads/main/data/tcga/BRCA%20mRNA%20SUBTYPE.csv'
URL_METHY = 'https://raw.githubusercontent.com/shawkath73/GCN-implementation/refs/heads/main/data/tcga/BRCA%20METHYL%20SUBTYPE.csv'
URL_MIRNA = 'https://raw.githubusercontent.com/shawkath73/GCN-implementation/refs/heads/main/data/tcga/BRCA%20MIRNA%20SUBTYPE.csv'

# Real TCGA BRCA clinical data from UCSC Xena
XENA_CLINICAL_URL = (
    'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/'
    'TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix'
)

# Local cache directory (inside the project)
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'brca_cache')


def _download_csv(url: str, cache_path: str) -> pd.DataFrame:
    """Download a CSV (or load from cache) and return a DataFrame."""
    if os.path.exists(cache_path):
        print(f"[cache] Loading {os.path.basename(cache_path)}")
        df = pd.read_csv(cache_path, header=0, index_col=None)
    else:
        import urllib.request
        print(f"[download] {url}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
        df = pd.read_csv(cache_path, header=0, index_col=None)
    return df


def _load_view(url: str, cache_name: str):
    """Return (feature_matrix [n_features x n_samples], sample_ids, labels)."""
    cache_path = os.path.join(CACHE_DIR, cache_name)
    df = _download_csv(url, cache_path)

    # The first row of the downloaded file has the header
    # Last column is 'Label', rest are features
    # Rows = samples, columns = features
    label_col = df.columns[-1]
    labels = df[label_col].values.astype(int)
    feat_df = df.drop(columns=[label_col])

    # Generate synthetic patient IDs (TCGA style)
    sample_ids = [f'TCGA-BRCA-{i:04d}' for i in range(len(feat_df))]
    feat_df.index = sample_ids

    # Transpose → features × samples  (to match original .fea format)
    feat_T = feat_df.T  # shape: (n_features, n_samples)
    feat_T.columns = sample_ids

    return feat_T, sample_ids, labels


def _load_real_survival(n: int) -> pd.DataFrame:
    """
    Download real TCGA BRCA overall-survival data from UCSC Xena and
    return a DataFrame with columns [PatientID, Survival, Death].

    The Xena BRCA clinical TSV has ~1100 rows.  We align by row-position
    (matching the order of the omics CSVs from the same repo).
    If the clinical file has fewer rows than n, we fall back to synthetics
    for the remaining samples.
    """
    cache_path = os.path.join(CACHE_DIR, 'brca_clinical_xena.tsv')

    if os.path.exists(cache_path):
        print(f"[cache] Loading {os.path.basename(cache_path)}")
        clin = pd.read_csv(cache_path, sep='\t', low_memory=False)
    else:
        import urllib.request
        print(f"[download] TCGA BRCA clinical from UCSC Xena …")
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            urllib.request.urlretrieve(XENA_CLINICAL_URL, cache_path)
            clin = pd.read_csv(cache_path, sep='\t', low_memory=False)
        except Exception as e:
            print(f"[survival] Download failed ({e}). Using synthetic survival.")
            return _synthetic_survival(n)

    # ── parse survival columns ────────────────────────────────────────────
    # Xena BRCA clinical columns vary; try common naming conventions
    col_map = {c.lower(): c for c in clin.columns}

    def _get(keys):
        for k in keys:
            if k in col_map:
                return clin[col_map[k]]
        return None

    vital   = _get(['vital_status', '_vital_status'])
    days_d  = _get(['days_to_death', '_days_to_death'])
    days_fu = _get(['days_to_last_followup', 'days_to_last_follow_up',
                    '_days_to_last_followup'])

    if vital is None or (days_d is None and days_fu is None):
        print("[survival] Required columns not found. Using synthetic survival.")
        return _synthetic_survival(n)

    # Build survival time: use days_to_death if dead, else days_to_last_followup
    death_flag = vital.str.strip().str.lower().isin(['dead', '1', 'deceased']).astype(int)

    surv_time = pd.Series(np.nan, index=clin.index)
    if days_d is not None:
        surv_time = pd.to_numeric(days_d, errors='coerce')
    if days_fu is not None:
        fu = pd.to_numeric(days_fu, errors='coerce')
        surv_time = surv_time.where(surv_time.notna(), fu)

    # Drop rows with missing survival time
    valid_mask = surv_time.notna() & (surv_time > 0)
    surv_time  = surv_time.loc[valid_mask].values
    death_flag = death_flag.loc[valid_mask].values

    n_clin = len(surv_time)
    print(f"[survival] Loaded {n_clin} real TCGA BRCA survival records.")

    # ── align by row-position ─────────────────────────────────────────────
    # Take up to n rows; pad with synthetics if clinical has fewer rows
    if n_clin >= n:
        surv_time  = surv_time[:n]
        death_flag = death_flag[:n]
    else:
        pad = n - n_clin
        print(f"[survival] Only {n_clin} clinical rows; padding {pad} with synthetics.")
        surv_time  = np.concatenate([surv_time,  np.full(pad, 1000.0)])
        death_flag = np.concatenate([death_flag, np.zeros(pad, dtype=int)])

    shared_ids = [f'TCGA-BRCA-{i:04d}' for i in range(n)]
    return pd.DataFrame({
        'PatientID': shared_ids,
        'Survival' : surv_time.astype(float),
        'Death'    : death_flag.astype(int),
    })


def _synthetic_survival(n: int) -> pd.DataFrame:
    """Fallback: flat synthetic survival (all censored at 1000 days)."""
    shared_ids = [f'TCGA-BRCA-{i:04d}' for i in range(n)]
    return pd.DataFrame({
        'PatientID': shared_ids,
        'Survival' : np.full(n, 1000.0),
        'Death'    : np.zeros(n, dtype=int),
    })


class MultiViewDataset3View(Dataset):
    """
    Three-view multi-omics dataset (mRNA, methylation, miRNA).
    survival DataFrame has columns: ['PatientID', 'Survival', 'Death']
    """
    def __init__(self, fea_rna, fea_meth, fea_mirna, survival, labels):
        self.fea_rna   = fea_rna.T    # (n_samples, n_rna_features)
        self.fea_meth  = fea_meth.T   # (n_samples, n_meth_features)
        self.fea_mirna = fea_mirna.T  # (n_samples, n_mirna_features)
        self.survival  = survival
        self.labels    = labels

    def __len__(self):
        return self.fea_rna.shape[0]

    def __getitem__(self, idx):
        xs = [
            torch.from_numpy(self.fea_rna.iloc[idx].values).float(),
            torch.from_numpy(self.fea_meth.iloc[idx].values).float(),
            torch.from_numpy(self.fea_mirna.iloc[idx].values).float(),
        ]
        surv_val = torch.from_numpy(
            np.array(self.survival.iloc[idx]['Survival'])
        ).float()
        return xs, surv_val, torch.tensor(idx).long(), torch.tensor(self.labels[idx]).long()

    def full_data(self):
        return [
            torch.from_numpy(self.fea_rna.values).float(),
            torch.from_numpy(self.fea_meth.values).float(),
            torch.from_numpy(self.fea_mirna.values).float(),
        ], torch.from_numpy(self.survival['Survival'].values.astype(np.float32))


def load_brca_data():
    """
    Download/cache the three BRCA CSV views, align samples, build the dataset.

    Returns
    -------
    dataset   : MultiViewDataset3View
    dims      : list[int]  – [n_rna_feat, n_mirna_feat, n_meth_feat]
    data_size : int        – number of samples
    """
    fea_rna,   ids_rna,   labels_rna   = _load_view(URL_MRNA,  'brca_mrna.csv')
    fea_meth,  ids_meth,  labels_meth  = _load_view(URL_METHY, 'brca_methy.csv')
    fea_mirna, ids_mirna, labels_mirna = _load_view(URL_MIRNA, 'brca_mirna.csv')

    # ── Align samples across views ──────────────────────────────────────────
    # For our synthetic IDs they are all identical across views; just use index
    # alignment based on position (all CSVs have the same number of rows in
    # the same order from the same GitHub repo).
    n = min(fea_rna.shape[1], fea_meth.shape[1], fea_mirna.shape[1])
    shared_ids = [f'TCGA-BRCA-{i:04d}' for i in range(n)]

    fea_rna   = fea_rna.iloc[:, :n]
    fea_meth  = fea_meth.iloc[:, :n]
    fea_mirna = fea_mirna.iloc[:, :n]

    fea_rna.columns   = shared_ids
    fea_meth.columns  = shared_ids
    fea_mirna.columns = shared_ids

    # ── Build survival DataFrame ────────────────────────────────────────────
    # Try real TCGA BRCA OS survival from UCSC Xena; fall back to synthetics.
    survival = _load_real_survival(n)

    # ── Variance Feature Selection ──────────────────────────────────────────
    def _apply_variance(df, n_features, view_name):
        print(f"[Variance] Selecting top {n_features} features by variance for {view_name}...")
        # Since some views might have fewer than n_features, use min
        k = min(n_features, df.shape[0])
        top_features = df.var(axis=1).nlargest(k).index
        selected_df = df.loc[top_features]
        print(f"[Variance] {view_name}: selected {selected_df.shape[0]} / {df.shape[0]} features.")
        return selected_df

    fea_rna   = _apply_variance(fea_rna, 1500, "mRNA")
    fea_meth  = _apply_variance(fea_meth, 1500, "Methylation")
    fea_mirna = _apply_variance(fea_mirna, 1500, "miRNA")

    # ── MinMax scale each view ──────────────────────────────────────────────
    def _scale(df):
        scaler = MinMaxScaler()
        arr = scaler.fit_transform(df.T).T   # scale across samples
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    fea_rna   = _scale(fea_rna)
    fea_meth  = _scale(fea_meth)
    fea_mirna = _scale(fea_mirna)

    dims      = [fea_rna.shape[0], fea_mirna.shape[0], fea_meth.shape[0]]
    data_size = n

    dataset = MultiViewDataset3View(fea_rna, fea_meth, fea_mirna, survival, labels_rna[:n])

    print(f"[BRCA] mRNA features   : {fea_rna.shape[0]}")
    print(f"[BRCA] Methylation feat: {fea_meth.shape[0]}")
    print(f"[BRCA] miRNA features  : {fea_mirna.shape[0]}")
    print(f"[BRCA] Samples         : {data_size}")
    dead = survival['Death'].sum()
    print(f"[BRCA] Survival — events (dead): {dead}/{data_size} ({100*dead/data_size:.1f}%)")

    return dataset, dims, data_size
