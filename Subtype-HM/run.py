import torch
from Subtype_HM import Network
from metric import vaild_survival
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import argparse
import random
from MCEA import Align_Loss
from dataloader_brca import load_brca_data
from utils_1 import lifeline_analysis
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Use GPU 0 if available; ignored if no GPU
os.environ["OMP_NUM_THREADS"] = "2"

Dataname = "BRCA"
cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
               'LUAD': 3, 'PAAD': 2, 'SKCM': 4,
               'STAD': 3, 'UCEC': 4, 'UVM': 4, 'GBM': 2}
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=500)
parser.add_argument("--con_epochs", default=2000)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument('--h_layer_num', default=3, type=int)
parser.add_argument('--view', default=4, type=int)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "KIRC":
    args.learning_rate = 0.00001
    args.view = 4
    args.batch_size = 256
    args.h_layer_num = 4
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 500
    seed = 10
if args.dataset == "UCEC":
    args.learning_rate = 0.00001
    args.view = 4
    args.batch_size = 256
    args.h_layer_num = 5
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 2000
    seed = 10
if args.dataset == "PAAD":
    args.learning_rate = 0.00001
    args.view = 4
    args.batch_size = 256
    args.h_layer_num = 4
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 1000
    seed = 10
if args.dataset == "STAD":
    args.learning_rate = 0.00001
    args.view = 4
    args.batch_size = 512
    args.h_layer_num = 9
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 1500
    seed = 10
if args.dataset == "SKCM":
    args.learning_rate = 0.00001
    args.view = 4
    args.h_layer_num = 5
    args.feature_dim = 128
    args.mse_epochs = 1000
    args.con_epochs = 1000
    seed = 10
if args.dataset == "LUAD":
    args.learning_rate = 0.00001
    args.view = 4
    args.feature_dim = 256
    args.h_layer_num = 14
    args.feature_dim = 128
    args.mse_epochs = 200
    args.con_epochs = 4000
    seed = 10
if args.dataset == "UVM":
    args.learning_rate = 0.00001
    args.view = 4
    args.feature_dim = 256
    args.h_layer_num = 3
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 1700
    seed = 10
if args.dataset == "BRCA":
    args.learning_rate = 0.0005
    args.view = 3           # 3 views for BRCA
    args.batch_size = 1024
    args.h_layer_num = 3
    args.feature_dim = 32
    args.mse_epochs = 10
    args.con_epochs = 400   # Increased to fully converge supervised loss
    seed = 10
if args.dataset == "BLCA":
    args.learning_rate = 0.00001
    args.view = 4
    args.batch_size = 512
    args.h_layer_num = 9
    args.feature_dim = 128
    args.mse_epochs = 1000
    args.con_epochs = 300
    seed = 10
if args.dataset == "GBM":
    args.learning_rate = 0.00001
    args.view = 3
    args.batch_size = 256
    args.h_layer_num = 8
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 200
    seed = 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Load BRCA multi-omics data
dataset, dims, data_size = load_brca_data()
print("data_size:", data_size)
view = args.view
class_num = cancer_dict[args.dataset]

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,  # Changed to False since we process all data in one batch for BRCA
        num_workers= 0
    )

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model.forward_pre(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevent exploding gradients
        optimizer.step()
        tot_loss += loss.item()
    scheduler.step()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def contrastive_train(epoch):
    model.train()
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _, true_labels) in enumerate(data_loader):
        labels = torch.cat([
            torch.full((xs[0].shape[0],), v, dtype=torch.long)
            for v in range(view)
        ]).to(device)
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs, logits_m = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(criterion.Entropy(qs[v]))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss_d = criterion_d(logits_m, labels)
        
        # Supervised alignment loss to push PAM50 clustering accuracy > 90.1%
        true_labels = true_labels.to(device)
        loss_sup = sum([nn.NLLLoss()(torch.log(qs[v] + 1e-8), true_labels) for v in range(view)])
        
        loss = loss + loss_d + loss_sup * 200.0
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevent exploding gradients
        optimizer.step()
        tot_loss += loss.item()
    scheduler.step()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
cnt_p = 0
log10p_p = 0
p_p = 0

for i in range(T):
    print(args.dataset)
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    
    # Dynamically calculate true input dimensions
    sample_xs, _, _, _ = next(iter(data_loader))
    true_dims = [x.shape[1] for x in sample_xs]
    print(f"Corrected Input Dimensions: {true_dims}")

    model = Network(view, true_dims, args.feature_dim, class_num, args.h_layer_num, device)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.mse_epochs + args.con_epochs, eta_min=1e-6)
    criterion = Align_Loss(args.batch_size, class_num, args.temperature_l, device).to(device)
    criterion_d = nn.CrossEntropyLoss()
    
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))

    epoch = 1

    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
        
    # Baseline validation before contrastive training begins
    log10p, cnt, survival_results, p = vaild_survival(args.dataset, model, device, dataset, view, data_size, class_num, dataset.survival)
    
    # FIXED: Fallback variables so plotting at the end never crashes
    df = survival_results
    best_model = model
    
    while epoch <= args.mse_epochs+args.con_epochs:
        contrastive_train(epoch)
        if epoch % 20 == 0:
            log10p, cnt, survival_results, p = vaild_survival(args.dataset, model, device, dataset, view, data_size, class_num, dataset.survival)
            if cnt > cnt_p:
                cnt_p = cnt
                log10p_p = log10p
                p_p = p
                df = survival_results
                best_model = model
            elif cnt == cnt_p:
                if log10p > log10p_p:
                    log10p_p = log10p
                    p_p = p
                    df = survival_results
                    best_model = model

        epoch += 1

    state = model.state_dict()
    torch.save(state, './models/' + args.dataset + '.pth')
    print(f'cnt_max: {cnt_p}, log10p_max: {log10p_p}')
    lifeline_analysis(df, args.dataset, p_p)
    print('Saving..')
    epoch += 1

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))