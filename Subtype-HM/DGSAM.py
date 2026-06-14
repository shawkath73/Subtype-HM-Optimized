import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


class DifferenceawareAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(DifferenceawareAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        # 共享的Q投影层
        self.shared_q_proj = nn.Linear(input_dim, input_dim)

        # 自注意力与交叉注意力的独立K/V投影
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 替换自注意力与交叉注意力的Q投影为共享参数
        self._init_shared_q_projection()

    def _init_shared_q_projection(self):
        """ 将共享的Q投影参数注入到自注意力和交叉注意力中 """
        # 获取共享Q的权重和偏置
        q_weight = self.shared_q_proj.weight  # (input_dim, input_dim)
        q_bias = self.shared_q_proj.bias  # (input_dim,)

        # 自注意力的参数替换
        with torch.no_grad():
            # 自注意力的in_proj_weight形状为 (3*input_dim, input_dim)
            # 前input_dim行是Q投影
            self.self_attn.in_proj_weight[:self.input_dim] = q_weight
            self.self_attn.in_proj_bias[:self.input_dim] = q_bias

            # 交叉注意力的Q投影同样替换
            self.cross_attn.in_proj_weight[:self.input_dim] = q_weight
            self.cross_attn.in_proj_bias[:self.input_dim] = q_bias

    def forward(self, query, key_list):
        """
        Args:
            query: 当前组学特征 (batch_size, input_dim)
            key_list: 其他组学特征列表，每个元素形状为 (batch_size, input_dim)
        Returns:
            融合后的特征 (batch_size, input_dim)
        """
        # 自注意力：共享Q投影已注入，直接使用原始输入
        self_attn_out, _ = self.self_attn(
            query.unsqueeze(1),  # (batch_size, 1, input_dim)
            query.unsqueeze(1),
            query.unsqueeze(1)
        )

        # 交叉注意力：计算组学差异作为键值
        keys = torch.stack(key_list, dim=1)  # (batch_size, num_keys, input_dim)
        differ_keys = query.unsqueeze(1) - keys

        cross_attn_out, _ = self.cross_attn(
            query.unsqueeze(1),  # 共享Q投影
            differ_keys,
            differ_keys
        )

        # 融合结果
        combined = self_attn_out + cross_attn_out
        return combined.squeeze(1)


# --- FULLY DYNAMIC CLASSIFIER ---
# This single class now handles 3 views, 4 views, or any number of views dynamically.
class MultiModalClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiModalClassifier, self).__init__()
        self.attention = DifferenceawareAttention(input_dim=input_dim, num_heads=4)
        self.classifier = nn.Linear(input_dim, num_classes)  # 分类层

    def forward(self, *Xs):
        """
        Dynamically handles any number of view inputs (e.g. X1, X2, X3 or X1, X2, X3, X4)
        """
        diffs = []
        # Loop through whatever number of views are passed
        for i in range(len(Xs)):
            # All other views become the key_list for the current query view
            other_Xs = list(Xs[:i] + Xs[i+1:])
            diff_i = self.attention(Xs[i], other_Xs)
            diffs.append(diff_i)

        # 拼接所有差异特征
        features = torch.cat(diffs, dim=0) 

        # 分类预测
        logits = self.classifier(features) 
        
        # Returns all individual diffs unpacked, followed by the logits
        return (*diffs, logits)


def train_one_epoch(model, optimizer, criterion, X1, X2, X3, X4, labels):
    model.train()
    optimizer.zero_grad()

    # 前向传播 (Dynamic call)
    diff_1, diff_2, diff_3, diff_4, logits = model(X1, X2, X3, X4)

    # 计算损失
    loss = criterion(logits, labels)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")


if __name__ == '__main__':
    # 模拟四个模态的数据 (batch_size=32, input_dim=128)
    X1 = torch.randn(32, 128)  # 模态 1
    X2 = torch.randn(32, 128)  # 模态 2
    X3 = torch.randn(32, 128)  # 模态 3
    X4 = torch.randn(32, 128)  # 模态 4
    
    # 定义标签：X1 -> 0, X2 -> 1, X3 -> 2, X4 -> 3
    labels = torch.cat([
        torch.zeros(32, dtype=torch.long),       # 标签 0
        torch.ones(32, dtype=torch.long),        # 标签 1
        torch.full((32,), 2, dtype=torch.long),  # 标签 2
        torch.full((32,), 3, dtype=torch.long)   # 标签 3
    ])

    # Instantiate the dynamic model with 4 classes
    model = MultiModalClassifier(input_dim=128, num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失

    # 训练一个 epoch
    for i in range(10):
        train_one_epoch(model, optimizer, criterion, X1, X2, X3, X4, labels)