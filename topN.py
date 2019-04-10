"""
主题模型推荐模块
使用计算topN
使用MRR来衡量一次查询的结果
"""

from db_connect import connect_mongodb_col
import theme_bot


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


def MRR(theme, k=1.0):
    """
    计算MRR值
    :param testing_dict:
    :return:
    """
    bot = theme_bot.ThemeBot(theme=theme)  # 创建主题机器人
    bot.start(train=True, k=k)
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


def main():
    theme = 'linux'
    # for i in range(10):
    mrr, testing_list, distribution = MRR(theme, 0.1)
    print(mrr)
    print(distribution)
    print('---------------------------------------------')


if __name__ == '__main__':
    main()
