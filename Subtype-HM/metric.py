from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from utils_1 import *
import os
os.environ["OMP_NUM_THREADS"] = "8"  # 控制线程数为4



def evaluate_survival(dataname,pred,survival):
    output = pd.DataFrame(columns=['sample_name', 'dcc'])
    sample_name = list(survival['PatientID'])
    output['sample_name'] = sample_name
    output['dcc'] = pred
    out_file = './results/' + dataname +'.dcc'
    output.to_csv(out_file, index=False, sep='\t')
    survival['label'] = pred
    try:
        res = log_rank(survival)
        p = res['p']
        log10p = res['log10p']
    except Exception as e:
        print(f"[log_rank] skipped ({e})")
        p, log10p = 1.0, 0.0

    clinical_path = "./data/TCGA_phenotype"
    if os.path.isdir(clinical_path):
        try:
            clinical = get_clinical(clinical_path, survival, dataname)
            cnt = clinical_enrichement(clinical, dataname)
        except Exception as e:
            print(f"[clinical] skipped: {e}")
            cnt = 0
    else:
        cnt = 0  # phenotype data not available locally

    return log10p,cnt,survival,p



def inference(loader, model, device, view, data_size):
    model.eval()
    soft_vector = []
    pred_vectors = []
    Hs = []
    Zs = []
    for v in range(view):
        pred_vectors.append([])
        Hs.append([])
        Zs.append([])
    labels_vector = []

    for step, (xs, y, _, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            qs, preds = model.forward_cluster(xs)
            hs, _, _, zs, _= model.forward(xs)
            q = sum(qs)/view
        for v in range(view):

            preds[v] = preds[v].detach()
            pred_vectors[v].extend(preds[v].cpu().detach().numpy())

        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())

    total_pred = np.argmax(np.array(soft_vector), axis=1)

    return total_pred, pred_vectors, Hs, labels_vector, Zs



#
def vaild_survival(dataname, model,device,dataset,view,data_size,class_num,survival,isprint=True):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False, num_workers=0
        )
    total_pred, pred_vectors, high_level_vectors, labels_vector, low_level_vectors = inference(test_loader, model, device, view, data_size)

    # print("Clustering results on cluster assignments of each view:")
    for v in range(view):
        log10p, cnt,survival_results,p = evaluate_survival(dataname=dataname, pred=pred_vectors[v], survival=survival)
        # print('view:{:.1f} log10p = {:.4f} cnt = {:.1f}'.format(v,log10p,cnt))

    # print("Clustering results on semantic labels: ")
    log10p, cnt, survival_results,p = evaluate_survival(dataname=dataname, pred=total_pred, survival=survival)
    # print('log10p = {:.4f} cnt = {:.1f}'.format(log10p, cnt))

    return log10p,cnt,survival_results,p
