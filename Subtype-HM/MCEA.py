import torch
import torch.nn as nn
import math


class Align_Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_l, device):
        super(Align_Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_label(self, q_i, q_j):

        q_i = q_i.t() + 1e-8  # Fix: Prevent empty cluster representations from becoming zero vectors (NaN norm)
        q_j = q_j.t() + 1e-8
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def Entropy(self, q):
        p = q.sum(0).view(-1)
        p /= p.sum()
        ne = math.log(p.size(0)) + (p * torch.log(p + 1e-8)).sum()
        return ne

