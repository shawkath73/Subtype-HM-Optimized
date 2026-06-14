import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from pytorch_kmeans import kmeans
from tqdm import tqdm
import pandas as pd
import torch


class HypergraphConstruction:
    def __init__(self, k=10):
        """
        :param view: 模态数量
        :param k: k-NN的k值
        """
        self.k = k

    def construct_instance_hypergraph(self, X_list):
        """
        构建实例超图关联矩阵 H_s，连接每个实例的不同模态顶点
        :param X_list: 包含每个模态数据的列表，每个元素大小为 (n_samples, n_features)
        :return: 实例超图关联矩阵 H_s
        """
        n_samples = X_list[0].shape[0]  # 假设每个模态的数据样本数量相同
        view = len(X_list)  # 模态数量
        # 初始化超图关联矩阵 H_s (在指定的设备上创建)
        H_s = torch.zeros((n_samples, view * n_samples), device=X_list[0].device)

        # 实例超边：连接同一个实例的不同模态
        for i in range(n_samples):
            for v in range(view):
                H_s[i, v * n_samples + i] = 1  # 对每个模态连接同一个实例

        return H_s

    def construct_modality_hypergraph(self, X, k):
        """
        构建模态超图关联矩阵 H_m，使用 k-NN 找到相似的顶点
        :param X: 模态数据，大小为 (n_samples, n_features)
        :param k: k-NN中的近邻数
        :return: 模态超图关联矩阵 H_m
        """
        n_samples = X.shape[0]
        # 计算所有样本之间的余弦相似度
        norms = torch.norm(X, dim=1, keepdim=True)
        X_normalized = X / (norms + 1e-8)  # 归一化向量
        similarity = torch.mm(X_normalized, X_normalized.t())  # 计算余弦相似度

        # 对角线上的值设置为 -inf，避免自身匹配
        similarity.fill_diagonal_(-float('inf'))

        # 找到每个样本的 k 个最近邻
        _, neighbors = similarity.topk(k, dim=1)

        # 初始化邻接矩阵 H_m
        H_m = torch.zeros((n_samples, n_samples), device=X.device)

        # 使用张量操作填充邻接矩阵
        hyper_indices = torch.arange(n_samples).unsqueeze(1).expand(-1, k).to(X.device)
        H_m[hyper_indices, neighbors] = 1

        return H_m


    def construct_cluster_hypergraph(self, X_list, cluster_num):
        n_samples = X_list[0].shape[0]  # 假设每个模态的数据样本数量相同
        view = len(X_list)  # 模态数量
        # 初始化超图关联矩阵 H_c (在指定的设备上创建)
        H_c = torch.zeros((view * cluster_num, view * n_samples), device=X_list[0].device)

        # 对每个模态的数据进行聚类
        cluster_labels = []
        for i in range(view):
            # 使用KMeans进行聚类
            cluster_ids, cluster_centers = kmeans(
                X=X_list[i].detach(), num_clusters=cluster_num, distance='euclidean', device=X_list[0].device
            )
            cluster_labels.append(cluster_ids)

        # 根据聚类结果构建超图关联矩阵 H_c
        for v in range(view):
            for i in range(n_samples):
                cluster_id = cluster_labels[v][i]  # 获取当前样本在模态 v 中的聚类标签
                # 计算当前样本在对应超边上的位置
                # 对于每个样本，设定对应聚类超边的位置为1
                H_c[v * cluster_num + cluster_id, v * n_samples + i] = 1

        return H_c



    def construct_full_hypergraph(self, X_list, cluster_num = 4):
        """
        构建完整超图：实例超图 + 模态超图
        :param X_list: 包含每个模态数据的列表
        :return: 实例超图 H_s 和 模态超图 H_m
        """
        n_samples = X_list[0].shape[0]

        # 1. 构建实例超图
        H_s = self.construct_instance_hypergraph(X_list)

        # 2. 分别构建每个模态的模态超图
        modality_hypergraphs = []
        for X in X_list:
            H_m = self.construct_modality_hypergraph(X, self.k)
            modality_hypergraphs.append(H_m)

        # 3. 构建完整的模态超图 H_m
        # 创建适当大小的零矩阵，并将模态超图块放入对角线上
        # 3. 在 GPU 上创建完整的模态超图 H_m
        zero_block = torch.zeros_like(modality_hypergraphs[0])

        H_m = torch.block_diag(*modality_hypergraphs).to(X_list[0].device)
        H_c = self.construct_cluster_hypergraph(X_list, cluster_num).to(X_list[0].device)
        return H_s, H_m, H_c

# 定义超图传播层
class HypergraphPropagation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HypergraphPropagation, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.theta)  # 初始化参数

    def forward(self, X, H):
        """
        :param X: 顶点特征矩阵 (n_samples, input_dim)
        :param H: 超图关联矩阵 (n_edges, n_samples)
        :return: 传播后的顶点特征 (n_samples, output_dim)
        """
        # 计算顶点度矩阵D
        D = torch.sum(H, dim=0)  # 按行求列得到度
        D_inv = torch.diag(1.0 / (D + 1e-8))  # D^(-1) with epsilon to prevent NaN

        # 计算超图传播
        H_T = H.t()  # 关联矩阵的转置
        propagated_features = D_inv @ H_T @ X @ self.theta
        return propagated_features

# 定义超图注意力层
class HypergraphAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dk):
        super(HypergraphAttention, self).__init__()
        self.W_q = nn.Parameter(torch.FloatTensor(input_dim, dk))
        self.W_k = nn.Parameter(torch.FloatTensor(input_dim, dk))
        self.W_v = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, X, Y, H):
        """
        :param X: 顶点特征矩阵 (n_samples, input_dim)
        :param Y: 超边特征矩阵 (n_edges, input_dim)
        :param H: 超图关联矩阵 (n_edges, n_samples)
        :return: 注意力加权后的顶点特征
        """
        # 计算注意力权重
        Q = Y @ self.W_q  # (n_edges, dk)
        K = X @ self.W_k  # (n_samples, dk)
        scaling_factor = torch.sqrt(torch.tensor(K.shape[1], dtype=torch.float32))

        # 计算注意力分数
        attention_scores = (Q @ K.t()) / scaling_factor  # (n_edges, n_samples)
        attention_scores = F.softmax(attention_scores.masked_fill(H.T == 0, float('-inf')), dim=-1)

        # 通过注意力加权更新顶点特征
        updated_features = attention_scores @ (X @ self.W_v)
        return updated_features


class HypergraphBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dk):
        super(HypergraphBlock, self).__init__()
        self.modality_propagation = HypergraphPropagation(input_dim, output_dim)
        self.cluster_propagation = HypergraphPropagation(input_dim, output_dim)
        self.attn_propagation = HypergraphAttention(hidden_dim, hidden_dim, output_dim, dk)

    def forward(self, X, Y_s_0, H_s, H_m,H_c):
        """
        :param X: 顶点特征矩阵 (n_samples, input_dim)
        :param Y_s_0: 初始实例超边特征
        :param H_s: 实例超图关联矩阵
        :param H_m: 模态超图关联矩阵
        :return: 更新后的顶点特征
        """
        Y_s_l = self.attn_propagation(X, Y_s_0, H_s.T)  # 实例超图传播
        Y_m_l = self.modality_propagation(X, H_m.T)  # 模态超图传播
        Y_c_l = self.cluster_propagation(X, H_c.T)  # 聚类超图传播


        # 连接实例,模态特征和聚类超边，并进行最终的传播
        X_updated = self.attn_propagation(
            torch.cat([Y_s_l, Y_m_l,Y_c_l], dim=0), X, torch.cat([H_s, H_m,H_c], dim=0)
        )
        return X_updated,Y_s_l,Y_m_l,Y_c_l


class HypergraphBlock_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dk, alpha=0.1):
        super(HypergraphBlock_2, self).__init__()
        self.modality_propagation = HypergraphPropagation(input_dim, output_dim)
        self.cluster_propagation = HypergraphPropagation(input_dim, output_dim)
        self.attn_propagation = HypergraphAttention(hidden_dim, hidden_dim, output_dim, dk)
        self.alpha = alpha

    def forward(self, X, Y_s_0, H_s, H_m, H_c):
        """
        :param X: 顶点特征矩阵 (n_samples, input_dim)
        :param Y_s_0: 初始实例超边特征
        :param H_s: 实例超图关联矩阵
        :param H_m: 模态超图关联矩阵
        :return: 更新后的顶点特征
        """
        Y_s_l = self.attn_propagation(X, Y_s_0, H_s.T)  # 实例超图传播
        Y_m_l = self.modality_propagation(X, H_m.T)  # 模态超图传播
        Y_c_l = self.cluster_propagation(X, H_c.T)  # 聚类超图传播
        # Prevent division by zero if H_m[0] is completely empty
        A = torch.matmul(H_c, H_m.T) / (torch.count_nonzero(H_m[0]) + 1e-8)
        Y_m_l = Y_m_l + self.alpha * torch.matmul(A.T,Y_c_l)

        # 连接实例和模态特征，并进行最终的传播
        X_updated = self.attn_propagation(
            torch.cat([Y_s_l, Y_m_l], dim=0), X, torch.cat([H_s, H_m], dim=0)
        )
        return X_updated, Y_s_l, Y_m_l, Y_c_l


class HypergraphPropagationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dk,layer_num = 3):
        super(HypergraphPropagationModule, self).__init__()
        self.layer_num = layer_num

        self.blocks = nn.ModuleList(
            [HypergraphBlock_2(input_dim, hidden_dim, output_dim, dk) for _ in range(layer_num)]
        )

    def forward(self,X, X_list, H_s, H_m,H_c):
        """
        :param X_list: 各模态特征列表，例如 [X_v, X_a, X_t, X_e]，每个元素为 (n_samples, input_dim)
        :param H_s: 实例超图关联矩阵
        :param H_m: 模态超图关联矩阵
        :return: 更新后的顶点特征
        """
        # 1. 计算初始实例超边特征（对所有模态的特征取平均）
        Y_s_0 = sum(X_list) / len(X_list)  # 计算所有模态的平均特征
        X_updated ,Y_s_l,Y_m_l,Y_c_l= self.blocks[0](X, Y_s_0, H_s, H_m,H_c)
        for i in range(1,self.layer_num):
            X_updated,Y_s_l,Y_m_l,Y_c_l = self.blocks[i](X_updated, Y_s_l, H_s, H_m,H_c)

        zh = torch.split(X_updated, X_list[0].shape[0], dim=0)
        # # 取平均
        # zh = sum(zh) / len(zh)

        return zh

if __name__ == '__main__':

    # 示例数据输入
    X_v = torch.from_numpy(np.random.rand(20, 128)).float() # 视频模态数据
    X_a = torch.from_numpy(np.random.rand(20, 128)).float()  # 音频模态数据
    X_t = torch.from_numpy(np.random.rand(20, 128)).float()  # 文本模态数据
    X_e = torch.from_numpy(np.random.rand(20, 128)).float()  # 额外模态数据

    X_list = [X_v, X_a, X_t, X_e]
    x_n = torch.cat(X_list, dim=0)
    print(X_list)
    knn_num = 5
    # 定义构建器，假设有3个模态
    hypergraph_constructor = HypergraphConstruction(k=knn_num)

    # 构建实例超图和模态超图
    H_s, H_m, H_c = hypergraph_constructor.construct_full_hypergraph(X_list,cluster_num=4)
    # H_c^T * H_m 计算模态超边和聚类超边的关联系数
    A = torch.matmul(H_c, H_m.T) / knn_num # (K, N) * (N, N) -> (K, N)

    print("实例超图 H_s 的形状：", H_s.shape)
    print("模态超图 H_m 的形状：", H_m.shape)
    print("聚类超图 H_c 的形状：", H_c.shape)
    # H_s = torch.from_numpy(H_s).float()


    model = HypergraphPropagationModule(input_dim=128, hidden_dim=128, output_dim=128, dk=32)

    X_updated = model(x_n,X_list, H_s, H_m,H_c)

    print("更新后的顶点特征形状：", X_updated[0].shape)


