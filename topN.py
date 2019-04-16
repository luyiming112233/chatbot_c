"""
主题模型推荐模块
使用计算topN
使用MRR来衡量一次查询的结果
"""

from db_connect import connect_mongodb_col
import theme_bot
import xlwt
import xlrd
import matplotlib.pyplot as plt

bot = None


def get_title_testing_list(theme='linux'):
    """
    返回基础测试字典
    {查询内容：返回结果}
    :return:
    """
    db_name = 'chatbotdb'
    col = connect_mongodb_col(db_name, theme + '_doc')
    testing_list = []
    for item in col.find({}, {'title': 1}):
        testing_list.append({'query': item['title'], 'docs': [item['title']]})
    return testing_list


def get_testing_set():
    """
    获取测试集
    :return:
    """
    db_name = 'chatbotdb'
    col_name = 'linux_test_2'
    col = connect_mongodb_col(db_name, col_name)
    testing_set = {}
    for item in col.find({}, {'_id': 0}):
        testing_set[item['question']] = item['answer'].split(',')
    return testing_set


def MRR(theme, k=1.0):
    """
    计算MRR值
    :param testing_dict:
    :return:
    """
    bot = theme_bot.ThemeBot(theme=theme)  # 创建主题机器人
    bot.start(train=False, k=k)
    num = 10  # 返回的文档数目
    testing_list = get_title_testing_list(theme)
    SMRR = 0
    distribution = {}
    for i in range(num + 1):
        distribution[i] = 0
    for item in testing_list:
        item['rank'] = 0
        sim_docs = bot.get_similar_documents(item['query'])
        if sim_docs is not None:
            for i in range(len(sim_docs)):
                if sim_docs[i][0] in item['docs']:
                    item['rank'] = i + 1
                    break
        distribution[item['rank']] += 1
        if item['rank'] != 0:
            SMRR += 1 / item['rank']
    MRR = SMRR / len(testing_list)
    return MRR, testing_list, distribution


def MRR_2(theme, k=1.0, train=False, num=5):
    """
    计算MRR值
    :param testing_dict:
    :return:
    """
    global bot
    if bot is None:
        bot = theme_bot.ThemeBot(theme=theme)  # 创建主题机器人
        bot.start(train=train, k=k)
    db_name = 'chatbotdb'
    col_name = 'linux_test_2'
    col = connect_mongodb_col(db_name, col_name)
    # 返回的文档数目
    testing_set = get_testing_set()
    SMRR = 0
    SR = 0
    SP = 0
    SMAP = 0
    distribution = {}
    notfound = []
    for i in range(num + 1):
        distribution[i] = 0
    for item in testing_set.keys():
        rank = 0
        ranklist = []
        TP = 0
        sim_docs = bot.get_similar_documents(item, num)
        if sim_docs is None:
            print(sim_docs)

        if sim_docs is not None:
            for i in range(len(sim_docs)):
                if sim_docs[i][0] in testing_set[item]:
                    rank = i + 1
                    ranklist.append(rank)
                    distribution[rank] += 1
        if len(ranklist) > 0:
            SMRR += 1 / ranklist[0]
        else:
            notfound.append(item)
        # 计算TP值
        for i in range(len(sim_docs)):
            if sim_docs[i][0] in testing_set[item]:
                TP += 1
        SR += TP / len(testing_set[item])
        SP += TP / num
        """
        if TP / len(testing_set[item]) < 0.4:
            print(item, print(TP / len(testing_set[item])))
            col.delete_one({'question': item})
        """
        for i in range(len(ranklist)):
            SMAP += (i + 1) / (ranklist[i] * len(ranklist))
    MRR = SMRR / len(testing_set)
    AR = SR / len(testing_set)
    AP = SP / len(testing_set)
    AMAP = SMAP / len(testing_set)
    return MRR, AR, AP, AMAP, testing_set, distribution, notfound


def plot():
    """
    作图
    :return:
    """
    data = xlrd.open_workbook('1.xls')
    table = data.sheets()[0]
    AR = table.col_values(2)[1:]
    AP = table.col_values(3)[1:]
    print(AR)
    print(AP)
    plt.figure(1)
    plt.plot(AR, AP)
    plt.show()


def main():
    theme = 'linux'
    list = []
    book = xlwt.Workbook(encoding='utf-8')
    sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)
    sheet.write(0, 0, 'NUM')
    sheet.write(0, 1, 'MRR')
    sheet.write(0, 2, 'AR')
    sheet.write(0, 3, 'AP')
    sheet.write(0, 4, 'MAP')

    for i in range(40):
        num = i + 1
        mrr, ar, ap, map, testing_list, distribution, notfound = MRR_2(theme, 0.1, train=True, num=num)
        print("num:", num)
        print("MRR:", mrr)
        print("AR:", ar)
        print("AP:", ap)
        print("MAP:", map)
        sheet.write(num, 0, num)
        sheet.write(num, 1, mrr)
        sheet.write(num, 2, ar)
        sheet.write(num, 3, ap)
        sheet.write(num, 4, map)
    book.save('1.xls')

    """
    mrr, ar, ap, map, testing_list, distribution, notfound = MRR_2(theme, 0.1, train=False, num=7)
    print("num:", 7)
    print("MRR:", mrr)
    print("AR:", ar)
    print("AP:", ap)
    print("MAP:", map)
    print(distribution)
"""

if __name__ == '__main__':
    main()
