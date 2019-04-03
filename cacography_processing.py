# coding=utf-8
import pycorrector
import re
import math

"""
错别字处理模块
"""
English_dictionary = {}  # 用于记录常见的英语单词


def distinguish_english(word):
    """
    区分word是中文单词还是英文单词
    :param word:
    :return:
    """
    a = re.match('[a-zA-Z]+', word)
    if a is not None and a.group() == word:
        return True
    else:
        return False


def load_english_dict():
    from db_connect import connect_mongodb_col
    dbname = 'chatbotdb'
    colname = 'idf_dict'
    col = connect_mongodb_col(dbname, colname)
    for item in col.find({}, {'_id': 0}):
        if distinguish_english(item['word']):
            English_dictionary[item['word']] = item['idf']


def wrong_word_recognition(word_list):
    """
    传入单词列表，进行异常字符识别
    :param word_list:
    :return:
    """
    if len(English_dictionary) is 0:
        load_english_dict()
    corr_word_list = []
    for word in word_list:
        if distinguish_english(word) is True:
            if word in English_dictionary.keys():
                corr_word_list.append(word)
            else:
                corr_word_list.append(english_word_correct(word))
        else:
            corr_word_list.append(pycorrector.correct(word)[0])
    return corr_word_list


def english_word_correct(word):
    """
    通过编辑距离校正错误的英语单词
    :param word:
    :return:
    """
    rec_word = ''
    min_dis = 99999999
    for eng in English_dictionary.keys():
        dis = word_similarity(word, eng)
        if dis < min_dis:
            min_dis = dis
            rec_word = eng
    return rec_word


def word_similarity(word1, word2):
    """
    计算两个单词之间的编辑距离
    :param word1:
    :param word2:
    :return:
    """
    if len(word1) == 0 | len(word2) == 0 | abs(len(word1) - len(word2) > 7): return 99999999
    juzhen = [[0 for j in range(len(word1) + 1)] for i in range(len(word2) + 1)]
    for i in range(len(word2) + 1):
        juzhen[i][0] = i
    for j in range(len(word1) + 1):
        juzhen[0][j] = j
    for i in range(1, len(word2) + 1):
        for j in range(1, len(word1) + 1):
            if word2[i - 1] == word1[j - 1]:
                cost = 0
            else:
                cost = 1
            juzhen[i][j] = min(juzhen[i - 1][j] + 1, juzhen[i][j - 1] + 1, juzhen[i - 1][j - 1] + cost)
    return juzhen[len(word2)][len(word1)]


if __name__ == '__main__':
    word_list = ['linx', '你好', '感帽', 'mysqa']
    print(word_list)
    corr_word_list = wrong_word_recognition(word_list)
    print(corr_word_list)
