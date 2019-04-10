# coding=utf-8
import time
import math
import jieba
import os
import numpy as np
from corpora_processing import extract_tf_idf, pre_process_cn, jieba_initialize, extract_key_words
from db_connect import *
from function import *
from gensim import corpora, models, similarities


class ThemeBot:
    root_directory = 'model/theme_models'
    db_name = 'chatbotdb'
    theme_col_name = 'themebot'
    idf_col_name = "idf_dict"

    def __int__(self):
        self.theme = None  # 文档机器人主题
        self.types = None  # 文档机器人类别
        self.document_table = None  # 文档表表名
        self.document_mat = None  # 文档相似矩阵名
        self.key_words = None  # 主题关键词
        self.documents = None  # 文档
        self.tfidf_location = None  # tfidf存储文件的位置
        self.lsi_location = None  # lsi模型存储文件的位置
        self.index_location = None  # index存储文件的位置
        self.dictionary_location = None  # dict存储文件的位置
        self.tfidf = None
        self.lsi = None
        self.index = None
        self.dictionary = None
        self.id2idf_dictionary = None
        self.sim_mat = None
        # kmeans_data 存储聚类模型的数据，分别是：
        # clusterID:类的编号
        # cluster_center:类的中心
        # recom_v：推荐集合的稀疏矩阵
        self.kmeans_data = None

    def __init__(self, theme):
        self.theme = theme  # 文档机器人主题
        self.types = None  # 文档机器人类别
        self.document_table = None  # 文档表表名
        self.document_mat = None  # 文档相似矩阵名
        self.key_words = None  # 主题关键词
        self.documents = None  # 文档
        self.tfidf_location = None  # tfidf存储文件的位置
        self.lsi_location = None  # lsi模型存储文件的位置
        self.index_location = None  # index存储文件的位置
        self.dictionary_location = None  # dict存储文件的位置
        self.tfidf = None
        self.lsi = None
        self.index = None
        self.dictionary = None
        self.id2idf_dictionary = None
        self.sim_mat = None
        # kmeans_data 存储聚类模型的数据，分别是：
        # clusterID:类的编号
        # cluster_center:类的中心
        # recom_v：推荐集合的稀疏矩阵
        self.kmeans_data = None

    def set_theme(self, theme):
        self.theme = theme

    def start(self, train=False, k=1):
        """
        ThemeBot启动函数
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
            self.reset_docID()  # 重置文档ID
            self.documents = self.read_document()  # 读入文档
            # 问答推荐模型训练
            train_corpus = [
                item['title'].lower() * math.ceil(len(item['content']) / len(item['title'] ) * k) + item[
                    'content'].lower()
                for item in self.documents]
            self.train_by_lsi(train_corpus)
            # 历史推荐模型训练
            his_train_corpus = [item['title'].lower() + item['content'].lower() for item in self.documents]
            self.train_simiarity_mat(his_train_corpus)

    def load_properties(self):
        """
        从数据表中读取ThemeBot的基础属性
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
            self.document_table = data['document_table']
            self.document_mat = data['document_mat']
            path = ThemeBot.root_directory + '/' + self.theme + '/'
            self.tfidf_location = path + self.theme + '_tfidf'
            self.lsi_location = path + self.theme + '_lsi'
            self.index_location = path + self.theme + '_index'
            self.dictionary_location = path + self.theme + '_dict'
            print('ThemeBot属性载入成功')

    def display_properties(self):
        """
        输出ThemeBot的基础属性
        :return:
        """
        print('Theme:', self.theme)
        print('types:', self.types)
        print('document_table:', self.document_table)
        print('document_mat:', self.document_mat)

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
            if os.path.exists(ThemeBot.root_directory) is False:
                os.makedirs(ThemeBot.root_directory)
                print(ThemeBot.root_directory, '创建成功')
            else:
                print(ThemeBot.root_directory, '已经存在')
            model_path = ThemeBot.root_directory + '/' + self.theme
            if os.path.exists(model_path) is False:
                os.makedirs(model_path)
                print(model_path, '创建成功')
            else:
                print(model_path, '已经存在')
            print('ThemeBot_', self.theme, '初始化成功')

    def reset_docID(self):
        """
        对document标注ID
        :return:
        """
        coldoc = connect_mongodb_col(ThemeBot.db_name, self.document_table)
        i = 0
        for c in coldoc.find({}, {'_id': 1}).sort('_id'):
            i = i + 1
            coldoc.update({'_id': c['_id']}, {"$set": {'docID': i}})

    def load_model(self):
        """
        载入模型
        :return:
        """
        self.tfidf = models.TfidfModel.load(self.tfidf_location)
        self.index = similarities.MatrixSimilarity.load(self.index_location)
        self.lsi = models.LsiModel.load(self.lsi_location)
        self.dictionary = corpora.Dictionary.load(self.dictionary_location)

        self.sim_mat = take_up(self.document_mat)
        start = time.clock()
        self.documents = self.read_document()
        elapsed = (time.clock() - start)
        print(self.theme, "Time used:", elapsed)

        self.kmeans_data = load_np_list(self.theme)

    def read_document(self, reload=False, num=0):
        """
        读取的问题,num 默认为0，如果为0，则载入所有记录
        :param num:
        :return:
        """
        # 若文档未读入或reload=True，从数据库重新读入数据
        if self.documents is None or reload is True:
            col = connect_mongodb_col(ThemeBot.db_name, self.document_table)
            self.documents = [item for item in col.find({}, {'title': 1, 'content': 1, 'docID': 1}).sort('docID')]
        if num == 0:
            return self.documents
        else:
            return self.documents[:num]

    def get_dict_id2idf(self, dictionary):
        """
        将gensim的字典形式转化为(id,idf)
        :param dictionary:
        :return:
        """
        word_id_dict = dictionary.token2id
        idf_dict = {}
        id_idf = {}
        col = connect_mongodb_col(ThemeBot.db_name, ThemeBot.idf_col_name)
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

    def train_by_lsi(self, train_documents):
        """
        使用LSI模型训练
        将训练后的模型存入root_directory下的theme文件夹
        :param train_documents:
        :return:
        """
        lib_texts = pre_process_cn(train_documents, True)

        dictionary = corpora.Dictionary(lib_texts)
        word_idf_dictionary, idfs = self.get_dict_id2idf(dictionary)

        # doc2bow(): 将collection words 转为词袋，用两元组(word_id, word_frequency)表示
        corpus = [dictionary.doc2bow(text) for text in lib_texts]

        tfidf = models.TfidfModel()
        tfidf.idfs = idfs
        corpus_tfidf = tfidf[corpus]

        # 训练topic数量为300的LSI模型
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
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

    def train_simiarity_mat(self, train_documents):
        """
        计算得到文档相似矩阵
        :return:
        """
        docs_num = len(train_documents)
        # 语料预处理
        lib_texts = pre_process_cn(train_documents, True)
        dictionary = corpora.Dictionary(lib_texts)
        word_idf_dictionary, idfs = self.get_dict_id2idf(dictionary)

        # doc2bow(): 将collection words 转为词袋，用两元组(word_id, word_frequency)表示
        corpus = [dictionary.doc2bow(text) for text in lib_texts]

        tfidf = models.TfidfModel()
        tfidf.idfs = idfs
        corpus_tfidf = tfidf[corpus]

        # 训练topic数量为300的LSI模型
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
        index = similarities.MatrixSimilarity(lsi[corpus])  # index 是 gensim.similarities.docsim.MatrixSimilarity 实例

        # index.save(index_location)
        sims = index[lsi[corpus]]

        sim_mat = np.zeros((docs_num, docs_num))

        for i in range(docs_num):
            for j in range(docs_num):
                sim_mat[i, j] = sims[i, j]
        self.sim_mat = sim_mat
        insert_np(self.document_mat, sim_mat)

    def get_similar_documents(self, target_document, document_num=10):
        """
        获取相似的文档/问题
        :param target_question: 查询的问题
        :param question_num: 返回的相似问题个数
        :return:
        """
        # target_text = pre_process_cn(target_document, low_freq_filter=False)
        if type(target_document) == 'str':
            target_document = [target_document]

        target_text = [item for item in jieba.cut(target_document)]

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

        document_list = []

        # 查看结果
        print("target:", target_document)
        # 返回前document_num个相似文档标题
        for i in range(document_num):
            document_list.append([self.documents[sort_sims[i][0]]['title'], str(sort_sims[i][1])])
            # print(self.lsi[sort_sims[i][0]])
        return document_list

    def get_similar_documents_by_word_list(self, word_list, document_num=10):
        """
        传入已经完成分词的单词列表
        :param target_question: 查询的问题
        :param question_num: 返回的相似问题个数
        :return:
        """
        target_text = word_list

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

        document_list = []

        # 返回前document_num个相似文档标题
        for i in range(document_num):
            document_list.append([self.documents[sort_sims[i][0]]['title'], str(sort_sims[i][1])])
        return document_list

    def get_recommanded(self, history):
        """
        返回这个主题的推荐集合
        :param history:
        :return:
        """
        # 历史文档，将在推荐集中去除
        history_docs = set(r['title'] for r in history)
        # 初始化相似文档字典
        doc_sim_dict = {}
        # 读取当前时间
        now = time.time()
        # 一天的时间戳值
        day_value = 86400
        # 加权求和
        for r in history:
            # 转为时间数组
            timeArray = time.strptime(r['time'], "%Y-%m-%d %H:%M:%S")
            # 转为时间戳
            timeStamp = int(time.mktime(timeArray))
            diff_value = math.floor((now - timeStamp) / day_value)
            # 计算时间衰退值
            decay = exponential_decay(diff_value)
            sim_list = self.get_similarity_docID(r['docID'], 20)
            for s in sim_list:
                doc_sim_dict[s[0]] = decay * doc_sim_dict.get(s[0], 0) + s[1]
        # 将文档权重排序
        sort_dsd = sorted(doc_sim_dict.items(), key=lambda x: x[1], reverse=True)
        recommanded = [item for item in sort_dsd if item[0] not in history_docs]
        return recommanded[0:20]

    def get_similarity_docID(self, docID, doc_num=20):
        """
        获得与当前文档相似的文档ID和相似度，并进行标准化
        :param docID:
        :param doc_num:
        :return:
        """
        m_sort = sorted(enumerate(self.sim_mat[docID - 1,]), key=lambda x: x[1])
        m_sort.sort(reverse=True, key=lambda x: x[1])
        sim_list = m_sort[1:doc_num + 1]
        sum = 0
        for d in sim_list:
            sum += d[1]
        stan_sim_list = [(d[0] + 1, d[1] / sum) for d in sim_list]  # 将权重标准化，使相似权重和为1;将矩阵索引转化为docID
        return stan_sim_list

    def get_historical_record(self, user='lym', record_num=50):
        """
        从数据库中读取user的数据
        基于内容相似得到推荐集
        :param userID:
        :return:
        """
        colname = 'history'
        col = connect_mongodb_col(ThemeBot.db_name, colname)
        history = [item for item in
                   col.find({'user': user, 'theme': self.theme}, {'_id': 0, 'user': 0}).sort('time', -1)]
        # 文档的ID表
        doc_id_dict = {}
        for item in self.documents:
            doc_id_dict[item['title']] = item['docID']
        for item in history:
            item['docID'] = doc_id_dict[item['title']]

        # 历史文档，将在推荐集中去除
        history_docs = set(r['title'] for r in history)
        print(history_docs)
        # 初始化相似文档字典
        doc_sim_dict = {}
        # 读取当前时间
        now = time.time()
        # 一天的时间戳值
        day_value = 86400
        # 加权求和
        for r in history:
            # 转为时间数组
            timeArray = time.strptime(r['time'], "%Y-%m-%d %H:%M:%S")
            # 转为时间戳
            timeStamp = int(time.mktime(timeArray))
            diff_value = math.floor((now - timeStamp) / day_value)
            # 计算时间衰退值
            decay = exponential_decay(diff_value)
            sim_list = self.get_similarity_docID(r['docID'], 20)
            for s in sim_list:
                doc_sim_dict[s[0]] = decay * doc_sim_dict.get(s[0], 0) + s[1]
        # 将文档权重排序
        sort_dsd = sorted(doc_sim_dict.items(), key=lambda x: x[1], reverse=True)
        recommanded = [(self.documents[item[0] - 1]['title'], item[1]) for item in sort_dsd if
                       item[0] not in history_docs]
        r = [self.documents[item[0] - 1]['docID'] for item in sort_dsd
             if item[0] not in history_docs]
        print(r[0:20])
        return recommanded[0:20]

    def historical_recommanded(self, history, record_num=10):
        """
        通过传入的history记录，基于内容相似得到推荐集
        :param: userID
        :return:
        """
        # 文档的ID表
        doc_id_dict = {}
        for item in self.documents:
            doc_id_dict[item['title']] = item['docID']
        for item in history:
            item['docID'] = doc_id_dict[item['title']]
        # 历史文档，将在推荐集中去除
        history_docs = set(r['title'] for r in history)
        print(history_docs)
        # 初始化相似文档字典
        doc_sim_dict = {}
        # 读取当前时间
        now = time.time()
        # 一天的时间戳值
        day_value = 86400
        # 加权求和
        for r in history:
            # 转为时间数组
            timeArray = time.strptime(r['time'], "%Y-%m-%d %H:%M:%S")
            # 转为时间戳
            timeStamp = int(time.mktime(timeArray))
            diff_value = math.floor((now - timeStamp) / day_value)
            # 计算时间衰退值
            decay = exponential_decay(diff_value)
            sim_list = self.get_similarity_docID(r['docID'], 20)
            for s in sim_list:
                doc_sim_dict[s[0]] = decay * doc_sim_dict.get(s[0], 0) + s[1]
        # 将文档权重排序
        sort_dsd = sorted(doc_sim_dict.items(), key=lambda x: x[1], reverse=True)
        recommanded = [(self.documents[item[0] - 1]['title'], self.theme, item[1]) for item in sort_dsd if
                       item[0] not in history_docs]
        r = [self.documents[item[0] - 1]['docID'] for item in sort_dsd
             if item[0] not in history_docs]
        return recommanded[0:record_num]

    def get_user_portrait(self, user='lym'):
        """
        根据历史记录获得用户画像
        :param userID:
        :return:
        """
        colname = 'history'
        col = connect_mongodb_col(ThemeBot.db_name, colname)
        history = [item for item in
                   col.find({'user': user, 'theme': self.theme}, {'_id': 0, 'user': 0}).sort('time', -1)]
        # 文档的ID表
        doc_id_dict = {}
        for item in self.documents:
            doc_id_dict[item['title']] = item['docID']
        for item in history:
            item['docID'] = doc_id_dict[item['title']]
        # 历史文档，将在推荐集中去除
        # history_docs = set(r['title'] for r in history)
        # 读取当前时间
        now = time.time()
        # 一天的时间戳值
        day_value = 86400
        # 主题特征列表
        theme_feature = []
        # 加权求和
        for r in history:
            timeArray = time.strptime(r['time'], "%Y-%m-%d %H:%M:%S")  # 转为时间数组
            timeStamp = int(time.mktime(timeArray))  # 转为时间戳
            diff_value = math.floor((now - timeStamp) / day_value)
            decay = exponential_decay(diff_value)  # 计算时间衰退值
            target_text = [item for item in jieba.cut(r['title'])]  # 将文档标题进行分词
            ml_bow = self.dictionary.doc2bow(target_text)  # 词袋处理
            if len(ml_bow) == 0:  # 若提取的关键词不再字典中，返回空
                continue
            ml_lsi = self.lsi[ml_bow]  # ml_lsi 形式如 (topic_id, topic_value)# 在上面选择的模型数据 lsi 中，获得其主题特征
            sort_ml_lsi = sorted(ml_lsi, key=lambda x: x[1], reverse=True)[0:10]  # 加入时间因子
            for t in sort_ml_lsi:
                theme_feature.append((t[0], t[1] * decay))
        topics = self.lsi.show_topics()  # 获得主题机器人所有的主题

        topic_dict = {}  # 获得主题集合
        for topic_id in theme_feature:
            topic_dict[topic_id[0]] = topic_dict.get(topic_id[0], 0) + topic_id[1]
        key_dict = {}
        for v in topic_dict:
            topic = topics[v]
            s = topic[1].split("+")
            for a in s:
                sa = a.replace(' ', '').replace('"', '').split("*")
                key_dict[sa[1]] = key_dict.get(sa[1], 0) + float(sa[0]) * topic_dict[v] * 10
        keys = {self.theme: 0}
        for item in key_dict:
            keys[self.theme] += key_dict[item]
            keys[item] = key_dict[item]
        keys[self.theme] /= 10
        print(keys)
        return keys

    def get_history_mat(self, history, doc_id_dict):
        """
        获得用户历史矩阵
        :return:
        """
        user_history_dict = {}
        for item in history:
            if item['user'] not in user_history_dict.keys():
                user_history_dict[item['user']] = []
            user_history_dict[item['user']].append(item)

        mat = np.zeros((len(user_history_dict), len(doc_id_dict)))
        i = 0
        for u in user_history_dict.keys():
            user_history = user_history_dict[u]
            for h in user_history:
                mat[i][doc_id_dict[h['title']] - 1] = h['preference']
            i += 1
        return mat

    def user_cf(self):
        """
        基于k-means算法的用户协同过滤
        :return:
        """
        from sklearn.cluster import KMeans
        # 最小访问数阈值
        min_num = 5
        # 获取历史数据
        colname = 'history'
        col = connect_mongodb_col(ThemeBot.db_name, colname)
        history = [item for item in
                   col.find({'theme': self.theme, 'type': 'documentation'}, {'_id': 0}).sort('user', -1)]
        # 建立用户ID表
        user_id_dict = {}
        users = [item for item in
                 col.find({'theme': self.theme, 'type': 'documentation'}, {'user': 1}).sort('user', -1).distinct(
                     'user')]

        # 文档的ID表
        doc_id_dict = {}
        for item in self.documents:
            doc_id_dict[item['title']] = item['docID']
        mat = self.get_history_mat(history, doc_id_dict)  # 返回相似矩阵
        # n_clusters参数指定分组数量，random_state = 1用来重现同样的结果
        kmeans_model = KMeans(n_clusters=15, random_state=1)
        # 通过fit_transform()方法来训练模型
        senator_distances = kmeans_model.fit_transform(mat)

        # 用户类别区分
        user_type = {}

        for i in range(len(kmeans_model.labels_)):
            tp = str(kmeans_model.labels_[i])
            if tp not in user_type.keys():
                user_type[tp] = set()
            user_type[tp].add(i)
        # 获得类别推荐文档集合

        recommanded_dict = {}  # 将一类的文档查看记录求和
        final_user_cf = []  # 记录有效类
        for tp in user_type:
            # 获得这一组的用户集合
            sum = np.zeros(len(doc_id_dict))
            user_set = user_type[tp]
            for user in user_set:
                sum = np.array(sum) + np.array(mat[user])
            recommanded_dict[tp] = sum
        # 删除上一个版本的聚类结果
        delete_np_list(self.theme)
        # 用于临时存储kmeans模型数据
        temp_data = []
        # 获得推荐集合
        id = 0
        for tp in recommanded_dict.keys():
            print(recommanded_dict[tp])
            if max(recommanded_dict[tp]) < min_num:
                continue
            else:
                id += 1
                # 将cluster_center转化为0-1向量
                center = kmeans_model.cluster_centers_[int(tp)]
                for c in center:
                    if c > 0.5:
                        c = 1
                    else:
                        c = 0
                temp_data.append({'clusterID': id, 'cluster_center': center,
                                  'recom_v': recommanded_dict[tp]})
                insert_np_list(self.theme, center, recommanded_dict[tp])
        # 将数据存入类中
        self.kmeans_data = temp_data
        print(self.theme, '用户聚类完成')

    def user_cf_recommanded(self, user):
        """
        基于改用户历史和已经获得的用户类，进行喜好相似度匹配
        :param user:
        :return:
        """
        colname = 'history'
        col = connect_mongodb_col(ThemeBot.db_name, colname)
        history = [item for item in
                   col.find({'user': user, 'theme': self.theme}, {'_id': 0, 'user': 0}).sort('time', -1)]
        # 文档的ID表
        doc_id_dict = {}
        for item in self.documents:
            doc_id_dict[item['title']] = item['docID']
        for item in history:
            item['docID'] = doc_id_dict[item['title']]
        # 历史文档，将在推荐集中去除
        history_docs = set(r['title'] for r in history)
        # 统计用户历史记录
        user_vector = np.zeros(len(doc_id_dict))
        for h in history:
            user_vector[doc_id_dict[h['title']] - 1] += h['preference']
        max_v = 0
        max_id = 0
        for k in self.kmeans_data:
            if max_id == 0:
                max_id = k['clusterID']
                max_v = cal_similar(k['cluster_center'], user_vector)
            elif cal_similar(k['cluster_center'], user_vector) > max_v:
                max_id = k['clusterID']
                max_v = cal_similar(k['cluster_center'], user_vector)
        recom_v = self.kmeans_data[max_id - 1]['recom_v']
        print(recom_v)
        mean = np.mean(recom_v)
        recom_list = {}
        keys_list = list(doc_id_dict.keys())
        for i in range(len(recom_v)):
            if recom_v[i] > mean:
                recom_list[keys_list[i + 1]] = recom_v[i]
        sort_list = sorted(recom_list, key=lambda x: x[1], reverse=True)
        recom = list(set(sort_list) - set(history_docs))
        print(recom)
        return recom


def main():
    bot = ThemeBot('linux')
    bot.start()
    # bot.get_user_portrait('lym')
    # bot.display()
    while True:
        a = input('->')
        similar_questions = bot.get_similar_documents(a)
        if similar_questions is not None:
            for l in similar_questions:
                print(l)


if __name__ == '__main__':
    main()
