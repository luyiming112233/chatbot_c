from db_connect import connect_mongodb_col
import math

"""
从数据库中查找各个主题的文档内容，并写入xls文件中中
"""


def get_theme_docs(theme):
    """
    传入主题，获取相应的文档内容
    :param theme:
    :return:
    """
    num = 20
    dbname = 'chatbotdb'
    doc_col = connect_mongodb_col(dbname, theme + '_doc')
    l = []
    for item in doc_col.find({}, {'_id': 0}):
        l.append(item['title'].replace('Q：', ''))
        l.append(item['type'])
        clen = len(item['content'])
        ct = min(math.floor(clen / num), 10)
        for i in range(ct):
            l.append(item['content'][i * num:i * num + num])
    ques_col = connect_mongodb_col(dbname, theme + '_ques')
    for item in ques_col.find({}, {'_id': 0}):
        l.append(item['question'].replace('Q：', ''))
        l.append(item['type'])
        qlen = len(item['answer'])
        qt = min(math.floor(qlen / num), 10)
        for i in range(qt):
            l.append(item['answer'][i * num:i * num + num])
    return list(set(l))


def read_themes():
    """
    从数据库中读入所有的主题
    :return:
    """
    dbname = 'chatbotdb'
    col_name = 'themebot'
    col = connect_mongodb_col(dbname, col_name)
    themes = {}
    i = 0
    for item in col.find({}, {'theme': 1}).sort('theme'):
        i += 1
        themes[item['theme']] = i
    return themes


def trans_data():
    """
    将每个主题的数据读入并转存按格式转存到xls中
    格式为：内容，themeid
    :return:
    """
    import xlwt
    themes = read_themes()
    data_list = {}
    for theme in themes.keys():
        theme_data = get_theme_docs(theme)
        data_list[theme] = theme_data
    for theme in themes.keys():
        # 创建工作簿（workbook）和工作表（sheet）
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("Sheet Name")
        sheet.write(0, 0, 'content')
        sheet.write(0, 1, 'label')
        num = 0
        # 向这个主题文件写入标签文件
        for d_theme in data_list:
            v = 0
            if d_theme == theme:
                v = 1
            # 取其中一个主题写入
            for item in data_list[d_theme]:
                num += 1
                sheet.write(num, 0, item)
                sheet.write(num, 1, v)
        workbook.save('data/{}.xls'.format(theme))


if __name__ == '__main__':
    trans_data()
