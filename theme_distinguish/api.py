# -*- coding: utf-8 -*-
"""
API
----
ALL application can be use
"""
from theme_distinguish.model import Interrogative, get_model
from theme_distinguish.get_data_from_db import read_themes

__all__ = ["train", "recognize"]

models = None


def train():
    """
    model training
    """
    model = get_model()
    model.train()


def recognize(sentence):
    """
    interrogative sentence recognize
    """
    model = get_model()
    prob = model.predict(sentence)[0]
    print(prob)
    return True if prob > 0.5 else False


def t_train():
    """
    对模型进行训练
    :return:
    """
    global models
    themes = read_themes()
    models = {}
    for theme in themes:
        model = Interrogative(theme)
        model.train()
        models[theme] = model


def predict(sentence):
    """
    进行主题分类
    :param sentence:
    :return:
    """
    global models
    if models is None:
        load_models()
    model_sim = {}
    for theme in models.keys():
        model_sim[theme] = models[theme].predict(sentence)[0]
    print(model_sim)
    return model_sim


def classify(sentence):
    """
    调用每个模型的predict，得到回归值
    根据回归值进行分类预测
    :param sentence:
    :return:
    """
    model_sim = predict(sentence)
    threshold = 0.5  # 判断属于该类的阈值
    possible_themes = []  # 超过阈值的主题
    most_possible_theme = None  # 最大相似度对应的主题
    max_sim = 0  # 最大相似度
    # 选取相似度高于阈值的主题，如果没有则返回相似度最高的主题
    for theme in model_sim.keys():
        if model_sim[theme] > threshold:
            possible_themes.append(theme)
        if model_sim[theme] > max_sim:
            most_possible_theme = theme
            max_sim = model_sim[theme]
    if len(possible_themes) != 0:
        return possible_themes
    else:
        return [most_possible_theme]


def load_models():
    global models
    models = {}
    themes = read_themes()
    for theme in themes:
        models[theme] = Interrogative(theme)
