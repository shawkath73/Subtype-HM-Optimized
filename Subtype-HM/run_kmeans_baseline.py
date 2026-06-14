"""
run_kmeans_baseline.py
Runs a standard KMeans clustering on the concatenated multi-omics features
and saves the predictions for comparison.
"""
import os
import torch
import pandas as pd
from sklearn.cluster import KMeans
from dataloader_brca import load_brca_data

def run_baseline():
    print("Loading data for KMeans baseline...")
    dataset, _, data_size = load_brca_data()
    xs, _ = dataset.full_data()
    X_concat = torch.cat(xs, dim=1).numpy()
    
    # We need to know the number of clusters.
    # The labels are in dataset.labels
    n_classes = len(set(dataset.labels))
    print(f"Running KMeans with {n_classes} clusters...")
    
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_concat)
    
    # Save to results/KMeans_baseline.dcc
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'KMeans_baseline.dcc')
    
    # The evaluate_brca.py script expects a TSV with column 'dcc'
    df_pred = pd.DataFrame({'dcc': kmeans_labels})
    df_pred.to_csv(out_path, sep='\t', index=False)
    print(f"Saved KMeans predictions to {out_path}")
    print("You can now run evaluate_brca.py to compare the metrics.")

if __name__ == '__main__':
    run_baseline()
