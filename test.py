# coding=utf-8
import random
import time
from chatbot import ChatBot

from db_connect import connect_mongodb_col

"""
小脚本
"""


def set_types():
    col = connect_mongodb_col('chatbotdb', 'document_type')
    types = ''
    for item in col.find({}, {'_id': 0}):
        types += item['type_s'] + ','
    connect_mongodb_col('chatbotdb', 'themebot').update({'theme': 'linux'}, {"$set": {"types": types}})
    for i in types.split(','):
        print(i)


def initialize_linux_doc():
    """
    初始化linux_doc表单数据
    :return:
    """
    doc_col = connect_mongodb_col('chatbotdb', 'document')
    linux_doc_col = connect_mongodb_col('chatbotdb', 'linux_doc')
    list = []
    for item in doc_col.find({}, {'_id': 0}):
        list.append({'title': item['title'], 'type': item['type_s'],
                     'path': item['path'], 'content': item['content'], 'access': item['access'],
                     'docID': item['docID']})
    linux_doc_col.insert_many(list)


def tan_xing(theme):
    doc_col = connect_mongodb_col('chatbotdb', theme)
    zhuanyou_doc_col = connect_mongodb_col('chatbotdb', theme + '_doc')
    list = []
    type_list = []
    title_set = set()
    for item in doc_col.find({}, {'_id': 0}):
        if len(item['body']) == 0 or item['question'] in title_set:
            continue
        list.append(
            {'title': item['question'], 'type': item['application'] + '-' + item['catalog'], 'path': 'null',
             'content': item['body'], 'access': 0})
        title_set.add(item['question'])
        type_list.append(item['application'] + '-' + item['catalog'])
    zhuanyou_doc_col.insert_many(list)
    type_set = set(type_list)
    s = ''
    for item in type_set:
        s += item + ','
    theme_col = connect_mongodb_col('chatbotdb', 'themebot')
    result = theme_col.find_one({'theme': theme})
    if result is None:
        theme_col.insert({'theme': theme, 'types': s, 'document_table': theme + '_doc', 'document_mat': theme + '_mat'})


def random_records(items, num=10):
    """
    随机生成记录
    :param items:随机的记录集合
    :param num:
    :return:
    """
    a1 = (2019, 1, 1, 0, 0, 0, 0, 0, 0)  # 设置开始日期时间元组（1976-01-01 00：00：00）
    a2 = (2019, 3, 14, 23, 59, 59, 0, 0, 0)  # 设置结束日期时间元组（1990-12-31 23：59：59）

    start = time.mktime(a1)  # 生成开始时间戳
    end = time.mktime(a2)  # 生成结束时间戳
    records = {}
    # 随机生成num个使用记录
    for i in range(num):
        t = random.randint(start, end)  # 在开始和结束时间戳中随机取出一个
        date_touple = time.localtime(t)  # 将时间戳生成时间元组
        date = time.strftime("%Y-%m-%d %H:%M:%S", date_touple)  # 将时间元组转成格式化字符串（1976-05-21）
        item = random.choice(items)
        records[date] = item
    return records


def simulated_historical_record(user='lym', theme='linux'):
    """
    用于模拟用户的历史数据并插入数据库
    :return:
    """
    dbname = 'chatbotdb'
    colname = 'history'
    document_items = [item['title'] for item in connect_mongodb_col(dbname, theme + '_doc').find({}, {'title': 1})]
    record = random_records(document_items, 20)
    col = connect_mongodb_col(dbname, colname)
    insert_list = []  # 插入多条数据的列表
    for r in record.keys():
        insert_list.append({'user': user, 'time': r, 'theme': theme, 'title': record[r]})
    col.insert_many(insert_list)


if __name__ == '__main__':
    chatbot = ChatBot()
    chatbot.retrain_all_bots()
