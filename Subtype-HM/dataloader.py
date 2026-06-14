from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import scipy.io
import torch




class MultiViewDataset(Dataset):
    def __init__(self, fea_rna, fea_meth, fea_mirna, fea_CN, survival):
        # 保存加载的特征和生存数据
        self.fea_rna = fea_rna.T  # 转置使样本为行
        self.fea_meth = fea_meth.T  # 转置使样本为行
        self.fea_mirna = fea_mirna.T  # 转置使样本为行
        self.fea_CN = fea_CN.T  # 转置使样本为行
        self.survival = survival

    def __len__(self):
        # 返回数据集中样本的数量
        return self.fea_rna.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引处的四个视图的特征及其对应的生存标签
        return [torch.from_numpy(self.fea_rna.iloc[idx].values).float(),
            torch.from_numpy(self.fea_meth.iloc[idx].values).float(),
            torch.from_numpy(self.fea_mirna.iloc[idx].values).float(),
            torch.from_numpy(self.fea_CN.iloc[idx].values).float()], \
           torch.from_numpy(np.array(self.survival.iloc[idx]['Survival'])).float(), \
           torch.tensor(idx).long()

    def full_data(self):
        # 返回整个数据集
        return [torch.from_numpy(self.fea_rna.values).float(), torch.from_numpy(self.fea_meth.values).float(),
                torch.from_numpy(self.fea_mirna.values).float(), torch.from_numpy(self.fea_CN.values).float()], \
               torch.from_numpy(self.survival['Survival'].values.astype(np.float32))
class MultiViewDataset_2(Dataset):
    def __init__(self, fea_rna, fea_meth, fea_CN, survival):
        # 保存加载的特征和生存数据
        self.fea_rna = fea_rna.T  # 转置使样本为行
        self.fea_meth = fea_meth.T  # 转置使样本为行
        self.fea_CN = fea_CN.T  # 转置使样本为行
        self.survival = survival

    def __len__(self):
        # 返回数据集中样本的数量
        return self.fea_rna.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引处的四个视图的特征及其对应的生存标签
        return [torch.from_numpy(self.fea_rna.iloc[idx].values).float(),
            torch.from_numpy(self.fea_meth.iloc[idx].values).float(),
            torch.from_numpy(self.fea_CN.iloc[idx].values).float()], \
           torch.from_numpy(np.array(self.survival.iloc[idx]['Survival'])).float(), \
           torch.tensor(idx).long()

    def full_data(self):
        # 返回整个数据集
        return [torch.from_numpy(self.fea_rna.values).float(), torch.from_numpy(self.fea_meth.values).float(),
                torch.from_numpy(self.fea_mirna.values).float(), torch.from_numpy(self.fea_CN.values).float()], \
               torch.from_numpy(self.survival['Survival'].values.astype(np.float32))


def load_data_fea(dataset,path):
    fea_CN_file = path + 'fea/' + dataset + '/CN.fea'
    fea_CN = pd.read_csv(fea_CN_file, header=0, index_col=0, sep=',')
    # print(fea_CN)
    # print(fea_CN.shape)
    name_list = fea_CN.columns

    fea_meth_file = path + 'fea/' + dataset + '/meth.fea'
    fea_meth = pd.read_csv(fea_meth_file, header=0, index_col=0, sep=',')
    # print(fea_meth)
    # print(fea_meth.shape)

    fea_mirna_file = path + 'fea/' + dataset + '/miRNA.fea'
    fea_mirna = pd.read_csv(fea_mirna_file, header=0, index_col=0, sep=',')
    # print(fea_mirna)
    # print(fea_mirna.shape)

    fea_rna_file = path + 'fea/' + dataset + '/rna.fea'
    fea_rna = pd.read_csv(fea_rna_file, header=0, index_col=0, sep=',')
    # print(fea_rna)
    # print(fea_rna.shape)





    survival_file = path + 'TCGA_survival/' + 'TCGA-' + dataset + '.survival.tsv'
    survivals = pd.read_csv(survival_file, sep='\t')
    survival = pd.DataFrame(columns=['PatientID', 'Survival', 'Death'])
    survival['PatientID'] = survivals['_PATIENT']
    survival['Survival'] = survivals['OS.time']
    survival['Death'] = survivals['OS']
    #PATIENT_ID一样的数据只保留第一个
    survival = survival.drop_duplicates(subset='PatientID', keep='first')
    # print(survival)
    # print(survival.shape)
    # 根据name_list筛选survival
    survival = survival[survival['PatientID'].isin(name_list)]
    # print(survival)
    # print(survival.shape)
    survival_list = list(survival['PatientID'])
    fea_CN = fea_CN[survival_list]
    fea_meth = fea_meth[survival_list]
    fea_mirna = fea_mirna[survival_list]
    fea_rna = fea_rna[survival_list]

    dim = [fea_rna.shape[0], fea_mirna.shape[0], fea_meth.shape[0], fea_CN.shape[0]]
    data_size = fea_rna.shape[1]

    dataset = MultiViewDataset(fea_rna, fea_mirna, fea_meth, fea_CN, survival)


    return dataset, dim,  data_size

