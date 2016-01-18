import numpy as np
import h5py
import re
import pandas as pd
from nltk.corpus import stopwords
import nltk.data
from gensim.models import word2vec
from sklearn import cross_validation

test_data = pd.read_csv("../data/atmosphere_train.csv", delimiter=",")

targets = []
wordvecs = []
max_N = 0

model = word2vec.Word2Vec.load("300features_40minwords_10context")
# feature size of word vectors
f_size = 300

FILE_NAME_PRE = '../data/prepped_train_X_word2vec_wordvecs_'

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def make_wordlist(text, remove_stops=False):
    """
    Function that cleans the movie description text. Removes
    non-alphabetical letters and optionally english stop words.
    """
    # first step is to remove non-alphabetical characters
    words = clean_str(text).split()

    if remove_stops:
        stops = set(stopwords.words("english"))
        return [w for w in words if not w in stops]

    return words


def get_feature_mat(words, f_size, model):
    out = []

    # internal word list of word2vec
    idx2words = set(model.index2word)

    s = filter(lambda e: e in idx2words, words)

    for w in s:
        out.append(model[w])

    return np.array(out)


def get_labels(labels):
    return [
        int("atmosphere_food_for_thought" in labels),
        int("atmosphere_funny" in labels),
        int("atmosphere_action" in labels),
        int("atmosphere_emotional" in labels),
        int("atmosphere_romantic" in labels),
        int("atmosphere_dark" in labels),
        int("atmosphere_brutal" in labels),
        int("atmosphere_thrilling" in labels)
    ]


for idx, row in test_data.iterrows():
    if idx > 1:
        words = make_wordlist(row["descr"])
        if len(words) > max_N:
            max_N = len(words)
        wordvecs.append(get_feature_mat(words, f_size, model))
        targets.append(get_labels(row["labels"].split(",")))


# pad rows with R^300 vectors of zero
for idx, w in enumerate(wordvecs):
    w = w.tolist()

    while len(w) < max_N:
        w.append(np.zeros(300))

    wordvecs[idx] = np.array(w)

assert(len(wordvecs) == len(targets))

# tensor = np.array(wordvecs)

rs = cross_validation.ShuffleSplit(len(targets), n_iter=10, test_size=0.1,
        random_state=0)

datasets = []

for train_index, test_index in rs:
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in train_index:
        x_train.append(wordvecs[i])
        y_train.append(targets[i])

    for i in test_index:
        x_test.append(wordvecs[i])
        y_test.append(targets[i])

    datasets.append([[x_train, y_train], [x_test, y_test]])


for idx, d in enumerate(datasets):
    h5f = h5py.File(FILE_NAME_PRE + str(idx) + '.h5', 'w')

    h5f.create_dataset('dataset_' + str(idx) + '_x_train', data=np.array(d[0][0]))
    h5f.create_dataset('dataset_' + str(idx) + '_y_train', data=np.array(d[0][1]))
    h5f.create_dataset('dataset_' + str(idx) + '_x_test', data=np.array(d[1][0]))
    h5f.create_dataset('dataset_' + str(idx) + '_y_test', data=np.array(d[1][1]))

    h5f.close()
