#Code borrowed from Arango et al (2019)
#https://github.com/aymeam/User_distribution_experiments
#Downloaded Glove from https://nlp.stanford.edu/projects/glove/
#Used: Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download):

import sys

import re

FLAGS = re.MULTILINE | re.DOTALL

import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils import np_utils

from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

from gensim.parsing.preprocessing import STOPWORDS

import codecs
import operator
import gensim, sklearn
from collections import defaultdict
from string import punctuation
import xgboost as xgb

#python lstm.py -f ~/DATASETS/glove-twitter/GENSIM.glove.twitter.27B.25d.txt -d 25 --tokenizer glove --loss categorical_crossentropy --optimizer adam --initialize-weights random --learn-embeddings --epochs 10 --batch-size 512

import gensim.downloader as api
import random
import math

def badjatiya(raw_data, labels):

    name = 'Badjatiya et al (2017)'

    lstm_trained_model, vocab = lstm(raw_data, labels)

    print('LSTM layer built')

    classifier, pre_processed_data = gbdt(raw_data, lstm_trained_model, vocab)

    return name, classifier, pre_processed_data


def lstm(raw_data, all_labels):

    np.random.seed(42)

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('Vectors/glove-twitter-25')

    tweets = select_tweets(raw_data, word2vec_model)

    EMBEDDING_DIM = 200
    vocab = gen_vocab(tweets)

    X, y = gen_sequence(tweets=tweets, vocab=vocab, all_labels=all_labels)

    MAX_SEQUENCE_LENGTH = max([len(x) for x in X])

    print('MAX_SEQUENCE_LENGTH', MAX_SEQUENCE_LENGTH)

    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    y = np.array(y)

    data, y = sklearn.utils.shuffle(data, y)

    assert (data.shape[1] == MAX_SEQUENCE_LENGTH)

    model = lstm_model(sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM, vocab=vocab)

    return train_LSTM(X=data, y=y, model=model, epochs=10, batch_size=128), vocab

def gbdt(raw_data, lstm_model, vocab):
    lstm_embeddings = lstm_model.layers[0].get_weights()[0]

    word2vec_model1 = lstm_embeddings

    # word2vec_model1 = word2vec_model1.reshape((word2vec_model1.shape[1], word2vec_model1.shape[2]))
    word2vec_model1 = word2vec_model1.reshape((word2vec_model1.shape[0], word2vec_model1.shape[1]))

    word2vec_model = {}

    for k, v in vocab.items():
        word2vec_model[k] = word2vec_model1[int(v)]

    tweets = select_tweets(raw_data, word2vec_model)

    X, Y = gen_data(tweets, word2vec_model, 200)

    model = xgb.XGBClassifier(nthread=-1)

    return model, X

def select_tweets(raw_data, word2vec_model):
    tweet_return = []
    for tweet, label in raw_data:
        _emb = 0
        words = glove_tokenize(tweet.lower())

        for w in words:
            if w in word2vec_model:
                _emb+=1
        if _emb:
            tweet_return.append({
                'text': tweet.lower(),
                'label': label
                })
    print('Tweets selected:', len(tweet_return))
    return tweet_return

def gen_vocab(tweets):
    # Processing
    vocab = {}
    vocab_index = 1
    for tweet in tweets:
        text = glove_tokenize(tweet['text'].lower())
        #SWP - Fixed bug here. Added space in join
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                vocab_index += 1
    vocab['UNK'] = len(vocab) + 1
    print(vocab['UNK'])
    return vocab

def gen_sequence(tweets, vocab, all_labels):
    X, y = [], []
    for tweet in tweets:
        text = glove_tokenize(tweet['text'].lower())
        #SWP - Fixed bug here. Added space in join
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        seq = []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        if len(seq) > 100:
            print('Appending sequence length', len(seq), 'text:', text)
            print(text)

        X.append(seq)
        y.append(all_labels.index(tweet['label']))
    return X, y

def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def lstm_model(sequence_length, embedding_dim, vocab):
    model = Sequential()
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=True))
    model.add(Dropout(0.25))
    model.add(LSTM(50))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_LSTM(X, y, model, epochs, batch_size):

    print('epochs:', epochs)
    print('batch_size:', batch_size)

    #cv_object = KFold(n_splits=10, shuffle=True, random_state=42)
    sentence_len = X.shape[1]
    print('sentence_len:', sentence_len)

    #for train_index, test_index in cv_object.split(X):

    shuffle_weights(model)

    #X_train, y_train = X[train_index], y[train_index]
    y = y.reshape((len(y), 1))
    X_temp = np.hstack((X, y))

    for epoch in range(epochs):

        batches = batch_gen(X_temp, batch_size)

        batch_no = 1

        for X_batch in batches:

            print('epoch:', epoch, 'batch:', batch_no)

            batch_no = batch_no + 1

            x = X_batch[:, :sentence_len]
            y_temp = X_batch[:, sentence_len]

            try:
                #y_temp = np_utils.to_categorical(y_temp, nb_classes=3)
                y_temp = np_utils.to_categorical(y_temp, num_classes=3)
            except Exception as e:
                print(e)
                print(y_temp)
            loss, acc = model.train_on_batch(x, y_temp, class_weight=None)

    return model

def batch_gen(X, batch_size):
    n_batches = X.shape[0] / float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0] / float(batch_size)) * batch_size
    n = 0
    for i in range(0, n_batches):
        if i < n_batches - 1:
            batch = X[i * batch_size:(i + 1) * batch_size, :]
            yield batch

        else:
            batch = X[end:, :]
            n += X[end:, :].shape[0]
            yield batch

def gen_data(tweets, word2vec_model, dimension):
    X, y = [], []
    for tweet in tweets:
        words = glove_tokenize(tweet['text'])
        emb = np.zeros(dimension)
        for word in words:
            try:
                emb += word2vec_model[word]
            except:
                pass
        emb /= len(words)
        X.append((emb, tweet['label']))
        y.append(tweet['label'])

    return X, y

def glove_tokenize(text):
    text = tokenize(text)
    text = ''.join([c for c in text if c not in punctuation])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return words

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = u"<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        #SWP: Fixes syntax error in re.split(ur"....
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()


