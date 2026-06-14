import os
import re
import torch

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
import difflib
import math
import os
import re
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal, chi2_contingency
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pair_confusion_matrix, accuracy_score
from matplotlib.lines import Line2D  # <-- 用于自定义图例句柄


def print_model(model):
    for name, parameters in model.named_parameters(): # if there is no parameter in the model, this will raise an error
        print(name, ':', parameters.size())

def print_model_weights(model):

    for param in model.parameters():
        print(torch.sum(param.grad.data))

def check_model_updates(model):
    for ae in model:
        print(torch.sum(ae.enc_1.weight))


# def save_model(opts, model, optimizer, current_epoch):
#     out = os.path.join(opts.model_path, "checkpoint_{}.tar".format(current_epoch))
#     state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
#     torch.save(state, out)



def lifeline_analysis(df, title_g="brca", p=None):
    """
    绘制 Kaplan-Meier 生存曲线，包含：
      - 多分组绘制
      - 中位生存时间虚线
      - 删失(censor)标记的小竖线
      - 可选 p 值标注（如果传入 log10p）

    df: 必须包含以下列：
        - 'Survival': 生存时间
        - 'Death':    是否死亡(0=未发生,1=发生)
        - 'label':    分组(如 0,1,2,...)
    title_g: 图标题
    log10p: 如果不为 None，则在图上标注 p 值；若为 None 则不标注。
    """

    groups = sorted(df["label"].unique())
    n_groups = len(groups)

    plt.figure(figsize=(6.5, 5))

    kmf = KaplanMeierFitter()
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    # 用来存放自定义图例句柄的列表
    custom_handles = []

    for i, group in enumerate(groups):
        mask = (df["label"] == group)
        kmf.fit(
            durations=df.loc[mask, "Survival"],
            event_observed=df.loc[mask, "Death"],
            label=f"cluster_{group}"
        )

        # 1) 禁用 lifelines 默认图例
        ax = kmf.plot(
            ci_show=False,
            color=colors[i % len(colors)],
            linewidth=2,
            show_censors=True,
            censor_styles={"marker": "|", "ms": 8},
            legend=False   # <-- 禁用自动图例
        )

        # 2) 构建自定义句柄(横线 + 竖线)
        from matplotlib.lines import Line2D
        handle = Line2D(
            [], [],
            color=colors[i % len(colors)],
            linewidth=2,
            marker='|',
            markersize=8,
            label=f"subtype_{group}"
        )
        custom_handles.append(handle)


    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")

    # 3) 使用自定义句柄 + 将图例放到图上方
    plt.legend(
        handles=custom_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),  # 调整这个 y 值可以更高/更低
        ncol=min(n_groups,4)                # 并排显示 n_groups 列
    )
    plt.title(title_g, y=1.16)

    # 如果外部已经算好 p 值，就在图上标注
    if p is not None:
        p_value = p
        if p_value < 1e-4:
            p_text = "p < 1e-4"
        else:
            p_text = f"p = {p_value:.4f}"
        plt.text(
            x=0.15,
            y=0.2,
            s=p_text,
            fontsize=12,
            bbox=dict(boxstyle="round", fc="white", ec="black")
        )
    ax = plt.gca()

    ax.set_ylim(0, 1.05)  # 统一所有图的 Y 轴范围

    plt.tight_layout()
    os.makedirs("./results", exist_ok=True)
    plt.savefig(f"./results/{title_g}.png", dpi=300)
    plt.show()



# 富集分析
def clinical_enrichement(clinical,cancer_type):
    cnt = 0
    # age 连续 使用KW检验
    # print(label,clinical)
    stat, p_value_age = kruskal(np.array(clinical["age"]), np.array(clinical["label"]))
    if p_value_age < 0.05:
        cnt += 1
        # print("---age---")
    # 其余离散 卡方检验
    if cancer_type == 'UCEC':
        stat_names = ["gender","stage"]
        for stat_name in stat_names:
            if stat_name in clinical:
                c_table = pd.crosstab(clinical[stat_name],clinical["label"],margins = True)
                stat, p_value_other, dof, expected = chi2_contingency(c_table)
                if p_value_other < 0.05:
                    cnt += 1
    elif cancer_type == 'GBM':
        stat_names = ["gender"]
        for stat_name in stat_names:
            if stat_name in clinical:
                c_table = pd.crosstab(clinical[stat_name],clinical["label"],margins = True)
                stat, p_value_other, dof, expected = chi2_contingency(c_table)
                if p_value_other < 0.05:
                    cnt += 1
    else:
        stat_names = ["gender","T","M","N","stage"]
        for stat_name in stat_names:
            if stat_name in clinical:
                c_table = pd.crosstab(clinical[stat_name],clinical["label"],margins = True)
                stat, p_value_other, dof, expected = chi2_contingency(c_table)
                if p_value_other < 0.05:
                    cnt += 1
                    # print(f"---{stat_name}---")
    return cnt


def log_rank(df):
    '''
    :param df: 传入生存数据
    拥有字段：label（预测对标签） Survival（生存时间） Death（是否死亡）
    :return: res 包含了p log2p log10p
    '''
    res = dict()
    results = multivariate_logrank_test(df['Survival'], df['label'], df['Death'])
    res['p'] = results.summary['p'].item()
    res['log10p'] = -math.log10(results.summary['p'].item())
    res['log2p'] = -math.log2(results.summary['p'].item())
    return res

def get_clinical(path,survival,cancer_type):
    clinical = pd.read_csv(f"{path}/{cancer_type}_phenotype.csv", low_memory=False)
    if cancer_type == 'UCEC':
        clinical = clinical[['patientID','age', 'patient.gender', 'patient.stage_event.clinical_stage']]
        # 重命名列名
        clinical = clinical.rename(columns={'patientID': 'PatientID', 'age': 'age', 'patient.stage_event.clinical_stage': 'stage'})
        survival = pd.merge(survival, clinical, on='PatientID', how='left')

    elif cancer_type == 'GBM':
        clinical = clinical[['patientID','years_to_birth', 'gender']]
        clinical = clinical.rename(columns={'patientID': 'PatientID', 'years_to_birth': 'age'})
        survival = pd.merge(survival, clinical, on='PatientID', how='left')

    else:
        clinical = clinical[['patientID', 'years_to_birth', 'gender', 'pathology_T_stage', 'pathology_M_stage', 'pathology_N_stage', 'pathologic_stage']]
        # 重命名列名
        clinical = clinical.rename(columns={'patientID': 'PatientID', 'years_to_birth': 'age', 'pathology_T_stage': 'T', 'pathology_M_stage': 'M', 'pathology_N_stage': 'N', 'pathologic_stage': 'stage'})
        # 将survival的PatientID和clinical的PatientID进行匹配
        survival = pd.merge(survival, clinical, on='PatientID', how='left')
    return survival.dropna(axis=0, how='any')





# if __name__ == '__main__':
#     test_path = "/homeb/hx/Codes/Subtype-DCC/results/"
#     clinical_path = "/homeb/hx/Dataset/Benchmark"
#     cancers = ['aml', 'breast', 'melanoma', 'liver', 'colon', 'kidney', 'gbm', 'ovarian', 'lung', 'sarcoma']
#     for dataset_name in cancers:
#         print(r'----------------------{}----------------------'.format(dataset_name))
#         test_file = os.path.join(test_path, dataset_name + '.dcc')
#         survival = load_survival(dataset_name, clinical_path)
#         dcc_df = pd.read_csv(test_file, sep='\t')
#         print(dcc_df)
#         print(survival)
#         df = survival
#         df['label'] = dcc_df['dcc'].values
#         res = log_rank(df)
#         log = res['log10p']
#         # max_label = dcc_df['dcc'].values
#         # survival["label"] = np.array(max_label)
#         # clinical_data = get_clinical(clinical_path+ "/clinical", survival, dataset_name)
#         # cnt = clinical_enrichement(clinical_data['label'], clinical_data)
#         print(r'{}: {:.1f}'.format(dataset_name, log))