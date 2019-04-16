# coding=utf-8
import re

import jieba
import jieba.analyse
import db_connect
import os
import math

"""
corpora_processing.py
语料处理模块
"""

corpus_directory = 'corpus'
idf_dict_path = corpus_directory + '/dict.txt'
stopword_path = corpus_directory + '/stopword.txt'
addword_path = corpus_directory + '/addword.txt'
glo_stopwords = None
glo_addwords = None


def jieba_initialize():
    """
    对jieba的语料库进行自定义处理
    :return:
    """
    jieba.analyse.set_idf_path(idf_dict_path)
    # jieba.load_userdict(addword_path)


def stopwordslist(filepath):
    """
    载入停用词list
    :param filepath:
    :return:
    """
    global glo_stopwords
    if glo_stopwords is None:
        print('载入停用词表')
        glo_stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return glo_stopwords


def produce_addwordlist(corpus):
    """
    创建附加词list
    :return:
    """
    addword = []
    for item in corpus:
        addword.append(jieba.analyse.extract_tags(item))
    addword = set(sum(addword, []))
    # 将数据写入文件中
    if os.path.exists(corpus_directory) is False:
        os.mkdir(corpus_directory)
    fw = open(addword_path, 'w', encoding='utf-8')
    for k in addword:
        fw.write(k + '\n')
    fw.close()


def load_addwordlist():
    """
    载入附加词
    :return:
    """
    jieba.load_userdict(addword_path)


def extract_tf_idf(row_corpus, db_name='chatbotdb', col_name='idf_dict'):
    """
    从语料中分词获得tf-idf字典
    :param row_corpus:
    :param db_name:
    :param col_name:
    :return:
    """
    stopwords = stopwordslist(stopword_path)  # 加载停用词的路径
    # 获得文档个数
    total = len(row_corpus)
    # 包含所有语料单词的字典
    all_dict = {}
    # 记录IDF值的字典
    idf_dict = {}
    for c in row_corpus:
        content = c.replace("\r", "").replace("\n", "").replace("\\r", "").replace("\\n", "")  # 删除换行和多余的空格
        content_seg = []

        for word in content.split():  # 按空格区分英语单词
            for cut_word in jieba.cut(word):  # 使用jieba进行中文分词
                content_seg.append(cut_word)  # 为文件内容分词

        temp_dict = {}
        for seg in content_seg:
            if seg not in stopwords:
                temp_dict[seg] = 1

        for key in temp_dict.keys():
            num = all_dict.get(key, 0)
            all_dict[key] = num + 1

    # 计算idf并存入字典
    import math

    for key in all_dict.keys():
        if all_dict[key] != 1:
            p = '%.10f' % (math.log10(total / (all_dict[key] + 1)))
            idf_dict[key] = p

    idf_list = []
    # 将数据写入文件中
    if os.path.exists(corpus_directory) is False:
        os.mkdir(corpus_directory)
    fw = open(idf_dict_path, 'w', encoding='utf-8')
    for k in idf_dict:
        if k != '\n':
            fw.write(k + ' ' + idf_dict[k] + '\n')
            idf_list.append({"word": k, "idf": idf_dict[k]})
    fw.close()
    # 将语料库数据存入mongodb中
    idf_col = db_connect.connect_mongodb_col(db_name, col_name)
    # idf_col.remove({})
    idf_col.insert_many(idf_list)
    print("语料库生成完成")


def pre_process_cn(courses, low_freq_filter=True, key_word_num=10):
    """
        简化的 中文+英文 预处理
        1.去掉停用词
        2.去掉标点符号
        3.去掉低频词
    """
    texts_tokenized = []
    for document in courses:
        texts_tokenized_tmp = []
        for word in document.split():
            # texts_tokenized_tmp += jieba.analyse.extract_tags(word, key_word_num, False,allowPOS=('eng', 'v', 'n', 'l', 'nz', 'vn', 'x'))
            texts_tokenized_tmp += jieba.analyse.extract_tags(word, key_word_num, False)

        texts_tokenized.append(texts_tokenized_tmp)

    texts_filtered_stopwords = texts_tokenized

    # 去除停用词
    stopwords = stopwordslist(stopword_path)  # 加载停用词的路径
    texts_filtered = [[word for word in document if not word in stopwords] for document in
                      texts_filtered_stopwords]

    # 去除过低频词
    if low_freq_filter:
        all_stems = sum(texts_filtered, [])
        stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
        texts = [[stem for stem in text if stem not in stems_once] for text in texts_filtered]
    else:
        texts = texts_filtered
    return texts


def extract_key_words(corpus):
    return jieba.analyse.extract_tags(corpus)


def sentence_similarity(str1, str2):
    """
    比较两个句子的相似度
    :param str1:
    :param str2:
    :return:
    """
    # 去除标点符号
    punctuations = [',', '.', ':', ';', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '，', '。', '：',
                    '；', '（', '）', '【', '】', ' ', '//', '_', '％', '"', '、']
    words1 = [item for item in jieba.cut(str1) if item not in punctuations]
    words2 = [item for item in jieba.cut(str2) if item not in punctuations]
    word_list = list(set(words1 + words2))
    vector1 = []
    vector2 = []
    for i in range(len(word_list)):
        if word_list[i] in words1:
            vector1.append(1)
        else:
            vector1.append(0)
        if word_list[i] in words2:
            vector2.append(1)
        else:
            vector2.append(0)
    part_up = 0.0
    for v1, v2 in zip(vector1, vector2):
        part_up += v1 * v2
    part_down = math.sqrt(sum(vector1) * sum(vector2))
    if part_down == 0:
        return 0.0
    else:
        return part_up / part_down


def extract_addword():
    """
    提取添加词
    :return:
    """
    col = db_connect.connect_mongodb_col('chatbotdb', 'idf_dict')
    f = open('corpus/addword.txt', 'w', encoding='utf-8')
    for item in col.find({}, {'_id': 0}):
        if float(item['idf']) < 1 and len(item['word']) > 1:
            f.write(item['word'] + '\n')

    f.close()


if __name__ == '__main__':
    extract_addword()
