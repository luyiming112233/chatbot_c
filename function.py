# coding=utf-8
import numpy as np


def exponential_decay(t, init=1, m=30, finish=0.2):
    """
    时间衰退函数
    :param t:
    :param init:
    :param m:
    :param finish:
    :return:
    """
    alpha = np.log(init / finish) / m
    l = - np.log(init) / alpha
    decay = np.exp(-alpha * (t + l))
    return decay


def cal_similar(v_1, v_2):
    """
    计算两个向量的相似度
    返回交集/并集的值
    :param v_1:
    :param v_2:
    :return:
    """
    num = min([len(v_1), len(v_2)])
    intersection = 0
    unionsection = 0
    for i in range(num):
        if v_1[i] > 0 and v_2[i] > 0:
            intersection += 1
        elif v_1[i] > 0 or v_2[i] > 0:
            unionsection += 1
    return intersection/unionsection
