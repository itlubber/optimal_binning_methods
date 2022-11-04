import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


def data_describe(data, var_name_bf, target, feature_type):
    """
    统计各取值的正负样本分布 [累计样本个数，正例样本个数，负例样本个数] 并排序
    :param data: DataFrame 输入数据
    :param var_name_bf: str 待分箱变量
    :param target: str 标签变量（y)
    :param feature_type: 特征的类型：0（连续） 1（离散）
    :return: DataFrame 排好序的各组中正负样本分布 count
    """
    # 统计待离散化变量的取值类型（string or digits)
    data_type = data[var_name_bf].apply(lambda x: type(x)).unique()
    var_type = True if str in data_type else False # 实际取值的类型：false(数字） true(字符）
    
    # 是否需要根据正例样本比重编码，True：需要，False：不需要
    #                   0（连续）    1（离散）
    #     false（数字）    0              0（离散有序）
    #     true（字符）     ×             1（离散无序）
    if feature_type == var_type:
        ratio_indicator = var_type
    elif feature_type == 1:
        ratio_indicator = 0
    elif feature_type == 0:
        raise("特征%s的类型为连续型，与其实际取值（%s）型不一致，请重新定义特征类型！！！" % (var_name_bf, data_type))

    # 统计各分箱（group）内正负样本分布[累计样本个数，正例样本个数，负例样本个数]
    count = pd.crosstab(data[var_name_bf], data[target])
    total = count.sum(axis=1)
    
    # 排序：离散变量按照pos_ratio排序，连续变量按照index排序
    if ratio_indicator:
        count['pos_ratio'] = count[count.columns[count.columns.values>0]].sum(axis=1) * 1.0 / total#？？？
        count = count.sort_values('pos_ratio') #离散变量按照pos_ratio排序
        count = count.drop(columns = ['pos_ratio'])
    else:
        count = count.sort_index() # 连续变量按照index排序
    return count, ratio_indicator


def calc_ks(count):
    a = count.cumsum(axis=0) / count.sum(axis=0)
    a = a.fillna(1)
    a = abs(a[0] - a[1])
    
    count = count.sort_index(ascending=False)
    b = count.cumsum(axis=0) / count.sum(axis=0)
    b = b.fillna(1)
    b = abs(b[0] - b[1])
    ks = [a.values[idx] + b.values[len(a.index) - 2 - idx] for idx in range(len(a.index) - 1)]
    return ks


def get_best_cutpoint(count):
    """
    根据指标计算最佳分割点
    :param count:
    :return:
    """
    entropy_list = calc_ks(count)
    
    intv = entropy_list.index(max(entropy_list))
    return intv


def best_ks_dsct(count, max_interval):
    """
    基于best_ks的特征离散化方法
    :param count: DataFrame 待分箱变量的分布统计
    :param max_interval: int 最大分箱数量
    :return: 分组信息（group）
    """
    group = count.index.values.reshape(1, -1).tolist()  # 初始分箱:所有取值视为一个分箱
    # 重复划分，直到KS的箱体数达到预设阈值。
    while len(group) < max_interval:
        group_intv = group[0] # 先进先出
        if len(group_intv) == 1:
            group.append(group[0])
            group.pop(0)
            continue
        
        # 选择最佳分箱点。
        count_intv = count[count.index.isin(group_intv)]
        intv = get_best_cutpoint(count_intv)
        cut_point = group_intv[intv]
        
        # 状态更新
        group.append(group_intv[0:intv + 1])
        group.append(group_intv[intv + 1:])
        group.pop(0)
    return group


def best_ks_binning(data, var_name, target="target", max_interval=6, feature_type=0):
    """
    基于best_ks的离散化方法
    :param data: DataFrame 原始输入数据
    :param var_name: str 待离散化变量
    :param target: str 离散化后的变量
    :param max_interval: int 最大分箱数量
    :param binning_method: string 分箱方法
    :return: 分组信息（group）
    """
    data = data[~data[var_name].isnull()].reset_index(drop=True)
    # 1. 初始化：将每个值视为一个箱体统计各取值的正负样本分布 & 从小到大排序
    count, var_type = data_describe(data, var_name, target, feature_type)
    
    group = best_ks_dsct(count, max_interval)
    group.sort()

    if not feature_type:
        group = [ele[-1] for ele in group] if len(group[0]) == 1 else [group[0][0]] + [ele[-1] for ele in group]
        group[0] = group[0] - 0.001 if group[0] == 0 else group[0] * (1 - 0.001)  # 包含最小值
        group[-1] = group[-1] + 0.001 if group[-1] == 0 else group[-1] * (1 + 0.001)  # 包含最大值
        group.append(np.nan)
    else:
        group.append([np.nan])
    
    return group
