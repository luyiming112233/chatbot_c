# coding=utf-8
import math
import threading
import jieba
import jieba.analyse
import time
from interrogative.api import *
from theme_distinguish import api
from function import exponential_decay
from db_connect import connect_mongodb_col
from corpora_processing import extract_tf_idf, produce_addwordlist, load_addwordlist, sentence_similarity
from theme_bot import ThemeBot
from theme_ques_bot import ThemeQuesBot
from cacography_processing import wrong_word_recognition


class ChatBot:
    """
    主机器人,负责调用管理主题机器人，使其完成任务
    """
    dbname = 'chatbotdb'
    themebot_col_name = 'themebot'
    userdict_path = 'corpus/dict.txt'

    def __init__(self):
        jieba.load_userdict(ChatBot.userdict_path)
        themebot_col = connect_mongodb_col(ChatBot.dbname, ChatBot.themebot_col_name)
        self.themes = []
        self.themebots = {}
        self.themequesbots = {}
        for item in themebot_col.find({}, {'_id': 0}):
            self.themes.append(item['theme'])
        load_addwordlist()

    def extract_dict(self):
        """
        读取所有theme对应的文档内容
        对内容进行分词并计算tf-idf值
        :return:
        """
        print('开始分词并计算tf-idf值')
        row_corpus = []
        for theme in self.themes:
            theme_col = connect_mongodb_col(ChatBot.dbname, theme + '_doc')
            for item in theme_col.find({}, {'title': 1, 'content': 1}):
                row_corpus.append(item['title'].lower() + item['content'].lower())
        extract_tf_idf(row_corpus)
        print('成功获得字典')

    def extract_addword(self):
        """
        选取所有theme对应的类别内容
        对类别进行分词得到附加词表
        并将存入tf_idf字典中
        :return:
        """
        types = []
        themebot_col = connect_mongodb_col(ChatBot.dbname, ChatBot.themebot_col_name)
        for item in themebot_col.find({}, {'_id': 0}):
            types.append(item['theme'] + item['types'])
        produce_addwordlist(types)
        print('附加词生成成功!')

    def theme_bot_start(self, theme, train=False):
        """
        启动一个主题机器人
        :param theme:
        :param train:
        :return:
        """
        if theme not in self.themes:
            print(theme + '_bot 不存在,启动失败!')
            return
        bot = ThemeBot(theme)
        bot.start(train)
        print(theme + '_bot 启动成功!')
        self.themebots[theme] = bot
        # return bot

    def theme_ques_bot_start(self, theme, train=False):
        """
        启动一个主题问题机器人
        :param theme:
        :param train:
        :return:
        """
        if theme not in self.themes:
            print(theme + '_bot 不存在,启动失败!')
            return
        bot = ThemeQuesBot(theme)
        bot.start(train)
        print(theme + '_ques_bot 启动成功!')
        self.themequesbots[theme] = bot
        # return bot

    def retrain_all_bots(self):
        """
        重新训练所有themebot
        :return:
        """
        # 重新生成语料
        self.extract_dict()
        temp_theme_bot = {}
        temp_ques_theme_bot = {}
        for theme in self.themes:
            bot = self.theme_bot_start(theme, train=True)
            temp_theme_bot[theme] = bot
            ques_bot = self.theme_ques_bot_start(theme, train=True)
            temp_ques_theme_bot[theme] = ques_bot
        self.themebots = temp_theme_bot
        self.themequesbots = temp_ques_theme_bot
        print('所有themebot训练成功!')

    def start_all_bots(self):
        """
        启动所有的主题机器人
        :return:
        """
        # 进程队列
        threads = []
        start = time.clock()
        for theme in self.themes:
            t = threading.Thread(target=self.theme_bot_start, args=(theme, False))
            threads.append(t)
            t = threading.Thread(target=self.theme_ques_bot_start, args=(theme, False))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print('所有themebot启动成功!')
        elapsed = (time.clock() - start)
        print("Time used:", elapsed)

    """
    # 上一版本的匹配主题机器人方案
    def similar_theme_matching(self, target, theme_num=3):
        \"""
        根据传入的语句进行分词，然后与各个主题机器人进行关键词比较
        选取更相似的前theme_num个主题机器人进行匹配
        :param tatget:
        :return:
        \"""
        target_key_words = jieba.cut(target)
        theme_dict = {}
        for theme in self.themes:
            num = 0
            for word in target_key_words:
                if word in self.themebots[theme].key_words:
                    num = num + 1
            if num != 0:
                theme_dict[theme] = num
        if len(theme_dict) == 0:
            return self.themes
        else:
            print(theme_dict)
            return theme_dict.keys()
    """

    def similar_theme_matching(self, target, theme_num=3):
        """
        使用基于提升树实现的接口来匹配相似主题
        :param target:
        :param theme_num:
        :return:
        """
        taglist = jieba.cut(target)
        s = ' '.join(taglist)
        return api.classify(s)

    def similar_documents(self, target, themes):  # 该接口需要调整
        """
        传入目标问题和主题名，调用对应机器人的相似文档匹配函数
        :param themes:
        :return:
        """
        # 判断是否是疑问句
        taglist = jieba.cut(target)
        tag = recognize(' '.join(taglist))

        # 将语句分词
        temp_target_text = [item for item in jieba.cut(target)]
        print('temp_target_text:', temp_target_text)
        # 进行错别字判断并纠正
        target_text = wrong_word_recognition(temp_target_text)
        print('target_text', target_text)
        if tag:
            docs = []
            ques_docs = []
            # 相似文档匹配
            for theme in themes:
                list = self.themebots[theme].get_similar_documents_by_word_list(target_text)
                if list is None:
                    continue
                for l in list:
                    if float(l[1]) > 0.3:
                        docs.append(l)
            # 相似问题匹配
            for theme in themes:
                list = self.themequesbots[theme].get_similar_questions_by_word_list(target)
                if list is None:
                    continue
                for l in list:
                    if float(l[1]) > 0.3:
                        ques_docs.append(l)
            for doc in docs:
                doc[1] = float(doc[1]) + sentence_similarity(target, doc[0])
            for ques_doc in ques_docs:
                ques_doc[1] = float(ques_doc[1]) + sentence_similarity(target, ques_doc[0])
            # 相似文档排序
            sort_docs = sorted(docs, key=lambda x: x[1], reverse=True)[:5]
            # 相似问题排序
            sort_ques_docs = sorted(ques_docs, key=lambda x: x[1], reverse=True)[:5]
            # 重新排列
            resort_docs = sorted(sort_ques_docs + sort_docs, key=lambda x: x[1], reverse=True)
            return resort_docs
        else:
            docs = []
            # 相似文档匹配
            for theme in themes:
                list = self.themebots[theme].get_similar_documents_by_word_list(target_text)
                if list is None:
                    continue
                for l in list:
                    if float(l[1]) > 0.3:
                        docs.append(l)
            for doc in docs:
                doc[1] = float(doc[1]) + sentence_similarity(target, doc[0])
            # 相似文档排序
            sort_docs = sorted(docs, key=lambda x: x[1], reverse=True)
            return sort_docs[:10]

    def similar_recommanded(self, user, recommended_num=20, theme_num=3):
        """
        传入用户名和主题名，获取其历史记录，获得历史权重值，筛选出比重高的主题
        将历史记录传入对应的ThemeBot，获得相似文档的推荐集合
        :param themes:
        :return:
        """
        colname = 'history'
        col = connect_mongodb_col(ChatBot.dbname, colname)
        history = [item for item in col.find({'user': user}, {'_id': 0, 'user': 0}).sort('time', -1).limit(50)]
        # 初始化历史权重字典
        weight_dict = {theme: 0 for theme in self.themes}
        # 初始化历史记录字典，用于将历史记录按theme分类
        record_dict = {theme: [] for theme in self.themes}
        # 读取当前时间
        now = time.time()
        # 一天的时间戳值
        day_value = 86400
        for item in history:
            # 转为时间数组
            timeArray = time.strptime(item['time'], "%Y-%m-%d %H:%M:%S")
            # 转为时间戳
            timeStamp = int(time.mktime(timeArray))
            diff_value = math.floor((now - timeStamp) / day_value)
            # 计算历史权重
            item['decay'] = exponential_decay(diff_value)
            weight_dict[item['theme']] += exponential_decay(diff_value)
            record_dict[item['theme']].append(item)
        print(weight_dict)
        theme_list = sorted(weight_dict, key=weight_dict.get, reverse=True)
        weight_sum = 0
        for theme in record_dict.keys():
            print(record_dict[theme])
        # 计算权重和
        for theme in theme_list:
            weight_sum += weight_dict[theme]
        # 获取推荐集合
        recommended_list = []
        for theme in theme_list:
            record_num = math.floor(recommended_num * weight_dict[theme] / weight_sum)
            if record_num > 0:
                recommended_list.append(self.themebots[theme].historical_recommanded(record_dict[theme], record_num))
        return recommended_list

    def cal_user_portrait(self, user):
        """
        分析用户画像，并存入数据库
        :param user:
        :return:
        """
        colname = 'history'
        col_portrait_name = 'user_portrait'
        col = connect_mongodb_col(ChatBot.dbname, colname)
        history = [item for item in col.find({'user': user}, {'_id': 0, 'user': 0}).sort('time', -1).limit(50)]
        # 初始化历史记录字典，用于将历史记录按theme分类
        record_dict = {theme: [] for theme in self.themes}
        # 主题集合
        theme_set = set()
        for item in history:
            record_dict[item['theme']].append(item)
            theme_set.add(item['theme'])
        keys = {}
        for theme in theme_set:
            dict = self.themebots[theme].get_user_portrait(user)
            for word in dict:
                keys[word] = keys.get(word, 0) + dict[word]
        print(keys)
        key_str = ""
        for w in keys:
            key_str += w + "::" + str(keys[w]) + "--"
        col = connect_mongodb_col(ChatBot.dbname, col_portrait_name)
        col.insert({'user': user, 'portrait': key_str})
        return keys

    def get_user_portrait(self, user):
        """
        获取用户画像
        :param user:
        :return:
        """
        col_portrait_name = 'user_portrait'
        col = connect_mongodb_col(ChatBot.dbname, col_portrait_name)
        result = [item for item in col.find({'user': user}, {'_id': 0})]
        print(result)
        if len(result) == 0:
            return self.cal_user_portrait(user)
        else:
            dict = {}
            for item in result[0]['portrait'].split("--"):
                v = item.split("::")
                if len(v) == 2:
                    dict[v[0]] = float(v[1])
            return dict


def main():
    chatbot = ChatBot()
    chatbot.start_all_bots()
    # chatbot.cal_user_portrait('lym')
    # a = chatbot.get_user_portrait('lym')
    # print(a)
    while (True):
        a = input('->')
        list = chatbot.similar_documents(a, chatbot.similar_theme_matching(a))
        for l in list:
            print(l)


if __name__ == '__main__':
    main()
