# -*- coding: utf-8 -*-
"""
CORPUS
-------
For corpus pre-process and features extraction
"""
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from theme_distinguish.config import get_config
from theme_distinguish.util import analyzer
from db_connect import connect_mongodb_col

__corpus = None


class Corpus:
    _config = get_config()

    @classmethod
    def read_corpus_from_file(cls, file_path):
        """
        read interrogative train data from local
        """
        a = pd.read_excel(file_path)
        print(type(a))
        print(a)
        return a

    @classmethod
    def read_corpus_from_mongo(cls):
        """
        read interrogative train data from local
        """
        col = connect_mongodb_col('chatbotdb', 'ques_classify')
        list = []
        for item in col.find({}, {'_id': 0}):
            list.append({'content': item['content'], 'label': item['type']})
        print(list)
        return list

    @classmethod
    def perform_word_segment(cls, corpus):
        """
        process word segmenting use jieba tokenizer
        """
        tokenizer = jieba.Tokenizer()
        corpus['tokens'] = corpus.content.apply(lambda x: list(tokenizer.cut(x)))
        return corpus

    @classmethod
    def feature_extract(cls, train, theme, tfidf_save=True):
        """
        feature engineering, extract Tf-idf feature
        """
        vectorizer = TfidfVectorizer(smooth_idf=True,
                                     analyzer=analyzer,
                                     ngram_range=(1, 1),
                                     min_df=1, norm='l1')
        sparse_vector = vectorizer.fit_transform(train.tokens.apply(lambda x: ' '.join(x)).tolist())
        label = train.label.tolist()

        # tf-idf vectorizer save
        if tfidf_save:
            joblib.dump(vectorizer, cls._config.get('interrogative', 'tfidf_vectorizer_path').format(theme))

        return sparse_vector, label

    @classmethod
    def generator(cls, theme):
        """
        pre-process corpus and extract features
        """
        corpus_path = cls._config.get('interrogative', 'corpus_path').format(theme)

        print('corpus_path', corpus_path)

        corpus = cls.read_corpus_from_file(corpus_path)
        # corpus = cls.read_corpus_from_mongo()
        train = cls.perform_word_segment(corpus)
        return cls.feature_extract(train, theme=theme)

    def __init__(self):
        raise NotImplementedError()


def get_corpus():
    """
    singleton object generator
    """
    global __corpus
    if not __corpus:
        __corpus = Corpus
    return __corpus


if __name__ == '__main__':
    get_corpus().generator()
