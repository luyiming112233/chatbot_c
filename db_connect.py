# coding=utf-8
import pymongo
from bson.binary import Binary
import pickle


def connect_mongodb_col(db_name, col_name, localclient=True):
    """
    传入数据库名和集合名，返回该集合的游标
    :param db_name:
    :param col_name:
    :return:
    """
    localclient = False
    if localclient == True:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client[db_name]
    else:
        client = pymongo.MongoClient("mongodb://134.175.111.57:27017/")
        db = client[db_name]
        if db_name == "chatbotdb":
            db.authenticate('ggfly', 'ggfly')
    col = db[col_name]
    return col


def insert_np(matname, mat):
    """
    插入矩阵
    :param matname:
    :param mat:
    :return:
    """
    mat = Binary(pickle.dumps(mat, protocol=2))
    dbname = 'chatbotdb'
    colname = 'matrix'
    col = connect_mongodb_col(dbname, colname)
    col.delete_many({'name': matname})
    col.insert({'name': matname, 'mat': mat})


def delete_np_list(theme):
    """
    删除对应主题的聚类数据
    :param theme:
    :return:
    """
    dbname = 'chatbotdb'
    colname = 'vector'
    col = connect_mongodb_col(dbname, colname)
    col.delete_many({'theme': theme})


def load_np_list(theme):
    """
    载入对应主题的聚类数据
    :param theme:
    :return:
    """
    dbname = 'chatbotdb'
    colname = 'vector'
    col = connect_mongodb_col(dbname, colname)
    kmeans_data = []
    id = 0
    for item in col.find({'theme': theme}, {'_id': 0}):
        id += 1
        kmeans_data.append({'clusterID': id, 'cluster_center': pickle.loads(item['cluster_center']),
                            'recom_v': pickle.loads(item['recom_v'])})
    return kmeans_data;


def insert_np_list(theme, cluster_center, recom_v):
    """
    存入对应主题的一条分类数据
    :param theme:
    :param cluster_center:
    :param recom_v:
    :return:
    """
    cluster_center = Binary(pickle.dumps(cluster_center, protocol=1))
    recom_v = Binary(pickle.dumps(recom_v, protocol=1))
    dbname = 'chatbotdb'
    colname = 'vector'
    col = connect_mongodb_col(dbname, colname)
    col.insert({'theme': theme, 'cluster_center': cluster_center, 'recom_v': recom_v})


def take_up(matname):
    """
    取出矩阵
    :param matname:
    :return:
    """
    dbname = 'chatbotdb'
    colname = 'matrix'
    col = connect_mongodb_col(dbname, colname)

    f = col.find_one({'name': matname}, {'mat': 1})
    return pickle.loads(f['mat'])


if __name__ == '__main__':
    load_np_list('弹性计算')
