# coding=utf-8
import flask
from flask import Flask, g, render_template, request
from chatbot import ChatBot
import json

__all__ = ['app']
app = Flask(__name__)

chatbot = None

chatbot =ChatBot()
chatbot.start_all_bots()
print('启动成功')

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/')
def index():
    return 'Hello World</br>'


@app.route('/start', methods=['GET'])
def chatbot_start():
    global chatbot
    chatbot =ChatBot()
    chatbot.start_all_bots()
    print('启动成功')
    return json.dumps('启动成功', ensure_ascii=False)


@app.route('/train', methods=['GET'])
def chatbot_train():
    global chatbot
    if chatbot is None:
        chatbot =ChatBot()
    chatbot.retrain_all_bots()
    print('训练成功')
    return json.dumps('训练成功', ensure_ascii=False)


@app.route('/similar_titles', methods=['GET'])
def get_similar_titles():
    global chatbot
    if chatbot is None:
        return json.dumps('no chatbot', ensure_ascii=False)
    target = request.args.get('question')
    print("target:", target)
    similar_questions = chatbot.similar_documents(target, chatbot.similar_theme_matching(target))
    print(similar_questions)
    return json.dumps(similar_questions, ensure_ascii=False)


@app.route('/similar_recommanded', methods=['GET'])
def get_similar_recommanded():
    global chatbot
    if chatbot is None:
        return json.dumps('no chatbot', ensure_ascii=False)
    user = request.args.get('user')
    print("target:", user)
    recommended_docs = chatbot.similar_recommanded(user)
    print(json.dumps(recommended_docs))
    return json.dumps(recommended_docs, ensure_ascii=False)


if __name__ == '__main__':
    app.run()