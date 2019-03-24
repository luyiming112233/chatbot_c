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
