# coding=utf-8
import random
import time
import jieba
from chatbot import ChatBot
from theme_distinguish import api
from interrogative.api import recognize

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


def simulated_pattern_historical_record(theme='弹性计算', kind_num=10):
    """
    用于模式模拟用户的历史数据并插入数据库
    :return:
    """
    dbname = 'chatbotdb'
    colname = 'history'
    document_items = [item['title'] for item in connect_mongodb_col(dbname, theme + '_doc').find({}, {'title': 1})]
    kind_list = []
    for i in range(kind_num):
        kind_list.append(random.randint(15, 30))
    user_num = sum(kind_list)
    id = 0
    type_num = 0
    # 历史记录数据表
    col = connect_mongodb_col(dbname, colname)
    insert_list = []  # 插入多条数据的列表
    for kind in kind_list:
        type_num += 1
        p_num = random.randint(10, 20)
        pattern_items = random.sample(document_items, p_num)
        left_document = list(set(document_items) - set(pattern_items))
        for i in range(kind):
            id += 1
            user_name = 'user_' + str(type_num) + '_' + str(id)
            print(user_name)
            # 获取模板数据
            pattern_record = random_records(pattern_items, random.randint(5, p_num + 5))
            # 获取随机数据
            ran_records = random_records(left_document, random.randint(5, 15))
            for r in pattern_record.keys():
                per = random.randint(0, 100)
                p = 0
                if per < 5:
                    p = 0  # 不满意
                elif per < 85:
                    p = 1  # 一般
                else:
                    p = 2  # 满意
                insert_list.append(
                    {'user': user_name, 'time': r, 'theme': theme, 'title': pattern_record[r], 'type': 'documentation',
                     'preference': p})
            for r in ran_records.keys():
                per = random.randint(0, 100)
                p = 0
                if per < 5:
                    p = 0  # 不满意
                elif per < 85:
                    p = 1  # 一般
                else:
                    p = 2  # 满意
                insert_list.append(
                    {'user': user_name, 'time': r, 'theme': theme, 'title': ran_records[r], 'type': 'documentation',
                     'preference': p})
    col.insert_many(insert_list)


def simulated_historical_record(user='lym', theme='弹性计算'):
    """
    用于随机模拟用户的历史数据并插入数据库
    :return:
    """
    dbname = 'chatbotdb'
    colname = 'history'
    document_items = [item['title'] for item in connect_mongodb_col(dbname, theme + '_doc').find({}, {'title': 1})]
    record = random_records(document_items, random.randint(5, 25))
    col = connect_mongodb_col(dbname, colname)
    insert_list = []  # 插入多条数据的列表
    for r in record.keys():
        per = random.randint(0, 100)
        p = 0
        if per < 5:
            p = 0  # 不满意
        elif per < 85:
            p = 1  # 一般
        else:
            p = 2  # 满意
        insert_list.append(
            {'user': user, 'time': r, 'theme': theme, 'title': record[r], 'type': 'documentation', 'preference': p})
    col.insert_many(insert_list)


def random_time():
    a1 = (2019, 1, 1, 0, 0, 0, 0, 0, 0)  # 设置开始日期时间元组（1976-01-01 00：00：00）
    a2 = (2019, 3, 14, 23, 59, 59, 0, 0, 0)  # 设置结束日期时间元组（1990-12-31 23：59：59）

    start = time.mktime(a1)  # 生成开始时间戳
    end = time.mktime(a2)  # 生成结束时间戳
    t = random.randint(start, end)  # 在开始和结束时间戳中随机取出一个
    date_touple = time.localtime(t)  # 将时间戳生成时间元组
    date = time.strftime("%Y-%m-%d %H:%M:%S", date_touple)  # 将时间元组转成格式化字符串（1976-05-21）
    return date


def reset_theme_table(theme):
    dbname = 'chatbotdb'
    colname = theme + '_doc'
    col = connect_mongodb_col(dbname, colname, False)
    for item in col.find({}, {'_id': 1, 'answer': 1}):
        pos = random.randint(20, 100)
        neg = random.randint(10, 100)
        acc = random.randint(pos + neg, 500)
        com = random.randint(acc * 2, acc * 10)
        col.update({'_id': item['_id']},
                   {"$set": {"positive": pos, "negative": neg,
                             "access": acc, 'recommended': com,
                             'update_time': random_time()}})


def get_themes():
    col = connect_mongodb_col('chatbotdb', 'themebot')
    return [item['theme'] for item in col.find({}, {'theme': 1})]


def get_dict():
    col = connect_mongodb_col('chatbotdb', 'user_portrait')
    t = [item for item in col.find({}, {'_id': 0})]
    t = t[0]
    print(t)
    dict = {}
    for item in t['portrait'].split("--"):
        v = item.split("::")
        if len(v) == 2:
            dict[v[0]] = float(v[1])
    print(dict)


def simulated_user_data():
    users = ['user_rand_' + str(i + 1) for i in range(40)]
    themes = get_themes()
    for user in users:
        print('simulate:', user)
        simulated_historical_record(user, theme='弹性计算')


def theme_type(target):
    taglist = jieba.cut(target)
    s = ' '.join(taglist)
    print(s)
    print(api.classify(s))


def main():
    target_dict = {
        '新安装的Linux系统，没有eth0': 'linux',
        'linux系统中如何进入目录': 'linux',
        '如何使用shell编写脚本': 'linux',
        'Block 大小是否存在限制？':'大数据',
        '数据库搭建': '未知',
        '为什么尽量使用内建算子，而不是自定义函数？':'大数据'
                   }

    for target in target_dict.keys():
        theme_type(target)


def tt():
    from interrogative import api as my_api
    my_api.train()


if __name__ == '__main__':
    tt()
    # api.t_train()
    #reset_theme_table('大数据')
    #taglist = jieba.cut('为什么尽量使用内建算子，而不是自定义函数？')
    #tag = recognize(' '.join(taglist))
