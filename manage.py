# -*- coding: utf-8 -*-
from interrogative.api import *
import jieba
from db_connect import connect_mongodb_col

#train()
'''
col = connect_mongodb_col('lym_test', 'corpus')
list = []
for q in col.find({}, {'_id':0}):
    list.append(q['content'])
for i in range(len(list)):
    tag = recognize(list[i])
    output = '是疑问句' if tag else '不是疑问句'
    print(output)
'''
str = '如何删除linux目录的文件？'
cent = ' '.join(list(jieba.cut(str)))
print(cent)
tag = recognize(cent)
output = '是疑问句' if tag else '不是疑问句'
print(output)
