import torch.nn as nn
from torch.nn.functional import normalize
import torch
from HIL import *
from DGSAM import *

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num, layer_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.layer_num = layer_num
        self.device = device
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.hypergraph_constructor = HypergraphConstruction(k=20)
        self.hypergraph = HypergraphPropagationModule(input_dim=feature_dim, hidden_dim=feature_dim, output_dim=feature_dim, dk=64, layer_num=self.layer_num)
        
        # This will dynamically take the view count (3 for BRCA)
        self.differgenerate = MultiModalClassifier(input_dim=feature_dim, num_classes=view)

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim*2, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view
        self.class_num = class_num

    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        zs_diff = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
        X = torch.cat(zs, dim=0)
        
        # Build instance, omics, and cluster hypergraphs
        H_s, H_o, H_c = self.hypergraph_constructor.construct_full_hypergraph(zs, cluster_num=self.class_num)
        X_updated = self.hypergraph(X, zs, H_s, H_o, H_c)
        
        # Fixed for 3 views
        differ_z1, differ_z2, differ_z3, logits_m = self.differgenerate(zs[0], zs[1], zs[2])
        zs_diff.append(differ_z1)
        zs_diff.append(differ_z2)
        zs_diff.append(differ_z3)
        
        for v in range(self.view):
            z = torch.cat((X_updated[v], zs_diff[v]), dim=1)
            t = zs[v]
            xr = self.decoders[v](t)
            q = self.label_contrastive_module(z)
            xrs.append(xr)
            qs.append(q)
        return hs, qs, xrs, zs, logits_m

    def forward_pre(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            zs.append(z)
            xrs.append(xr)
        return hs, qs, xrs, zs

    def forward_cluster(self, xs):
        qs = []
        zs = []
        preds = []
        zs_diff = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
        X = torch.cat(zs, dim=0)
        
        # Build instance, omics, and cluster hypergraphs
        H_s, H_o, H_c = self.hypergraph_constructor.construct_full_hypergraph(zs, cluster_num=self.class_num)
        X_updated = self.hypergraph(X, zs, H_s, H_o, H_c)
        
        # Fixed for 3 views
        differ_z1, differ_z2, differ_z3, logits_m = self.differgenerate(zs[0], zs[1], zs[2])
        zs_diff.append(differ_z1)
        zs_diff.append(differ_z2)
        zs_diff.append(differ_z3)
        
        for v in range(self.view):
            z = torch.cat((X_updated[v], zs_diff[v]), dim=1)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)

        return qs, preds