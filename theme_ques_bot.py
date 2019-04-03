# coding=utf-8
import time
import math
import jieba
import os
import numpy as np
from corpora_processing import extract_tf_idf, pre_process_cn, jieba_initialize,extract_key_words
from db_connect import connect_mongodb_col, insert_np, take_up
from function import exponential_decay
from gensim import corpora, models, similarities


class ThemeQuesBot:
    root_directory = 'model/theme_ques_models'
    db_name = 'chatbotdb'
    theme_col_name = 'ThemeQuesBot'
    idf_col_name = "idf_dict"

    def __int__(self):
        self.theme = None  # 问题机器人主题
        self.types = None  # 问题机器人类别
        self.question_table = None  # 问题表表名
        self.key_words = None  # 问题关键词
        self.questions = None  # 问题
        self.tfidf_location = None  # tfidf存储文件的位置
        self.lsi_location = None  # lsi模型存储文件的位置
        self.index_location = None  # index存储文件的位置
        self.dictionary_location = None  # dict存储文件的位置
        self.tfidf = None
        self.lsi = None
        self.index = None
        self.dictionary = None
        self.id2idf_dictionary = None

    def __init__(self, theme):
        self.theme = theme  # 问题机器人主题
        self.types = None  # 问题机器人类别
        self.question_table = None  # 问题表表名
        self.key_words = None  # 问题关键词
        self.questions = None  # 问题
        self.tfidf_location = None  # tfidf存储文件的位置
        self.lsi_location = None  # lsi模型存储文件的位置
        self.index_location = None  # index存储文件的位置
        self.dictionary_location = None  # dict存储文件的位置
        self.tfidf = None
        self.lsi = None
        self.index = None
        self.dictionary = None
        self.id2idf_dictionary = None

    def set_theme(self, theme):
        self.theme = theme

    def start(self, train=False):
        """
        ThemeQuesBot启动函数
        train为False时，载入训练好的模型
        train为True时，重新训练模型
        :param train:
        :return:
        """
        if self.theme is None:
            print('Theme 属性值为空,机器人启动失败!')
            return
        jieba_initialize()
        if train is False:
            self.load_properties()  # 载入基础参数
            self.load_model()  # 载入模型和文档数组
        else:
            self.initialize()  # 初始化文件目录和数据库内容
            self.load_properties()  # 载入基础参数和文档数组
            self.questions = self.read_question()  # 读入文档
            # 问答推荐模型训练
            train_corpus = [
                item['question'].lower() * math.ceil(len(item['answer']) / len(item['question'])) + item['answer'].lower()
                for item in self.questions]
            self.train_by_lsi(train_corpus)


    def load_properties(self):
        """
        从数据表中读取ThemeQuesBot的基础属性
        :return:
        """
        if self.theme is None:
            print('Theme 属性值为空，载入属性失败!')
            return
        else:
            data = connect_mongodb_col('chatbotdb', 'themebot').find_one({'theme': self.theme}, {'_id': 0})
            self.types = data['types'].split(',')
            # 通过对theme和types分词得到该主题的关键词
            self.key_words = extract_key_words(self.theme.lower() + data['types'].lower())
            self.question_table = self.theme+'_ques'
            path = ThemeQuesBot.root_directory + '/' + self.theme + '/'
            self.tfidf_location = path + self.theme + '_tfidf'
            self.lsi_location = path + self.theme + '_lsi'
            self.index_location = path + self.theme + '_index'
            self.dictionary_location = path + self.theme + '_dict'
            print('ThemeQuesBot属性载入成功')


    def display_properties(self):
        """
        输出ThemeQuesBot的基础属性
        :return:
        """
        print('Theme:', self.theme)
        print('types:', self.types)
        print('question_table:', self.question_table)
        print('question_mat:', self.question_mat)

    def initialize(self):
        """
        创建新的主题机器人之前先进行初始化
        在root_directory下创建模型文件夹
        并在数据库中插入初始数据
        :return:
        """
        if self.theme is None:
            print('Theme is none , initialize fail !')
        else:
            # 基础文件夹创建
            if os.path.exists(ThemeQuesBot.root_directory) is False:
                os.makedirs(ThemeQuesBot.root_directory)
                print(ThemeQuesBot.root_directory, '创建成功')
            else:
                print(ThemeQuesBot.root_directory, '已经存在')
            model_path = ThemeQuesBot.root_directory + '/' + self.theme
            if os.path.exists(model_path) is False:
                os.makedirs(model_path)
                print(model_path, '创建成功')
            else:
                print(model_path, '已经存在')
            print('ThemeQuesBot_', self.theme, '初始化成功')

    def load_model(self):
        """
        载入模型
        :return:
        """
        self.tfidf = models.TfidfModel.load(self.tfidf_location)
        self.index = similarities.MatrixSimilarity.load(self.index_location)
        self.lsi = models.LsiModel.load(self.lsi_location)
        self.dictionary = corpora.Dictionary.load(self.dictionary_location)
        self.questions = self.read_question()

    def read_question(self, reload=False, num=0):
        """
        读取的问题,num 默认为0，如果为0，则载入所有问题记录
        :param num:
        :return:
        """
        # 若文档未读入或reload=True，从数据库重新读入数据
        if self.questions is None or reload is True:
            col = connect_mongodb_col(ThemeQuesBot.db_name, self.question_table)
            self.questions = [item for item in col.find({}, {'question': 1, 'answer': 1}).sort('question')]
        if num == 0:
            return self.questions
        else:
            return self.questions[:num]

    def get_dict_id2idf(self, dictionary):
        """
        将gensim的字典形式转化为(id,idf)
        :param dictionary:
        :return:
        """
        word_id_dict = dictionary.token2id
        idf_dict = {}
        id_idf = {}
        col = connect_mongodb_col(ThemeQuesBot.db_name, ThemeQuesBot.idf_col_name)
        id = 0
        for w in word_id_dict.keys():
            c = col.find_one({"word": w}, {"_id": 0, "word": 1, "idf": 1})
            if c is None:
                idf_dict[w] = float(1)
                id_idf[id] = float(1)
            else:
                idf_dict[w] = float(c["idf"])
                id_idf[id] = float(c["idf"])
            id = id + 1
        return idf_dict, id_idf

    def train_by_lsi(self, train_questions):
        """
        使用LSI模型训练
        将训练后的模型存入root_directory下的theme文件夹
        :param train_questions:
        :return:
        """
        lib_texts = pre_process_cn(train_questions, True)

        dictionary = corpora.Dictionary(lib_texts)
        word_idf_dictionary, idfs = self.get_dict_id2idf(dictionary)

        # doc2bow(): 将collection words 转为词袋，用两元组(word_id, word_frequency)表示
        corpus = [dictionary.doc2bow(text) for text in lib_texts]

        tfidf = models.TfidfModel()
        tfidf.idfs = idfs
        corpus_tfidf = tfidf[corpus]

        # 训练topic数量为num_topics的LSI模型
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50)
        index = similarities.MatrixSimilarity(lsi[corpus])  # index 是 gensim.similarities.docsim.MatrixSimilarity 实例

        # 存储模型的内容
        tfidf.save(self.tfidf_location)
        lsi.save(self.lsi_location)
        index.save(self.index_location)
        dictionary.save(self.dictionary_location)
        # 重新更新模型
        self.tfidf = tfidf
        self.index = index
        self.lsi = lsi
        self.dictionary = dictionary
        self.id2idf_dictionary = self.get_dict_id2idf(self.dictionary)
        print("模型训练结束")

    def get_similar_questions(self, target_question, question_num=10):
        """
        获取相似的文档/问题
        :param target_question: 查询的问题
        :param question_num: 返回的相似问题个数
        :return:
        """
        target_question = target_question
        # target_text = pre_process_cn(target_question, low_freq_filter=False)
        target_text = [item for item in jieba.cut(target_question)]

        print("target_text:", target_text)

        # 词袋处理
        ml_bow = self.dictionary.doc2bow(target_text)

        # 若提取的关键词不再字典中，返回空
        if len(ml_bow) == 0:
            return None

        for w in self.tfidf[ml_bow]:
            print(w[0], w[1])

        # 在上面选择的模型数据 lsi 中，计算其他数据与其的相似度
        ml_lsi = self.lsi[ml_bow]  # ml_lsi 形式如 (topic_id, topic_value)
        sims = self.index[ml_lsi]
        # 排序
        sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

        question_list = []

        # 查看结果
        print("target:", target_question)
        # 返回前question_num个相似文档标题
        for i in range(question_num):
            question_list.append([self.questions[sort_sims[i][0]]['question'], str(sort_sims[i][1])])
        return question_list

    def get_similar_questions_by_word_list(self, word_list, question_num=10):
        """
        获取相似的文档/问题
        :param target_question: 查询的问题
        :param question_num: 返回的相似问题个数
        :return:
        """

        target_text = word_list

        print("target_text:", target_text)

        # 词袋处理
        ml_bow = self.dictionary.doc2bow(target_text)

        # 若提取的关键词不再字典中，返回空
        if len(ml_bow) == 0:
            return None

        for w in self.tfidf[ml_bow]:
            print(w[0], w[1])

        # 在上面选择的模型数据 lsi 中，计算其他数据与其的相似度
        ml_lsi = self.lsi[ml_bow]  # ml_lsi 形式如 (topic_id, topic_value)
        sims = self.index[ml_lsi]
        # 排序
        sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

        question_list = []

        # 查看结果
        # 返回前question_num个相似文档标题
        for i in range(question_num):
            question_list.append([self.questions[sort_sims[i][0]]['question'], str(sort_sims[i][1])])
        return question_list


def main():
    bot = ThemeQuesBot('大数据')
    bot.start()
    #records = bot.get_historical_record()
    #print(records)
    while True:
        list = bot.get_similar_questions(input('->'))
        if list is None:
            print('None')
        else:
            for item in list:
                print(item)


if __name__ == '__main__':
    main()
