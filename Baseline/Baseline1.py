# coding: utf-8

# **TODO:**
# * bag of means (word2vec)
# * label distribution, c_hat vs c
# * corpus on all movie descriptions vs only the test set (needed for on the fly classification)
# * test distinct pred_labels for specific movie in test data

import re
from functools import partial
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier # binary relevance
from skmultilearn.meta.br import BinaryRelevance
from skmultilearn.meta.lp import LabelPowerset

# these are the metrics we want to use for evaluation
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

# actual estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB


# scoring metrics used for evaluation. namely precision, accuracy, hamming loss (recall)
# and f_1-score with several different averages
METRICS = ['precision_macro', 'recall_macro', 'f1_macro', 'precision_micro', 'recall_micro', 'f1_micro']

# ## Step 1: Prepare Training Data
data_csv = {
    'prepped_train_X_Doc2Vec_dbow_d100_n5_mc2_t2':'prepped_train_X_Doc2Vec_dbow_d100_n5_mc2_t2.csv',
    'prepped_train_X_Doc2Vec_dm_c_d100_n5_w5_mc2_t2': 'prepped_train_X_Doc2Vec_dm_c_d100_n5_w5_mc2_t2.csv',
    'prepped_train_X_Doc2Vec_dm_m_d100_n5_w10_mc2_t2': 'prepped_train_X_Doc2Vec_dm_m_d100_n5_w10_mc2_t2.csv',
    'prepped_train_X_Doc2Vec_dbow_dmc': 'prepped_train_X_dbow_dmc.csv',
    'prepped_train_X_Doc2Vec_dbow_dmm': 'prepped_train_X_dbow_dmm.csv',
    'prepped_train_X_bow_tfidf': 'prepped_train_X_bow_tfidf.csv',
    'prepped_train_X_bigrams': 'prepped_train_X_bigrams.csv',
    'prepped_train_X_trigrams': 'prepped_train_X_trigrams.csv',
    'prepped_train_X_bigrams_tfidf': 'prepped_train_X_bigrams_tfidf.csv',
    'prepped_train_X_trigrams_tfidf': 'prepped_train_X_trigrams_tfidf.csv',
    'prepped_train_X_bow': 'prepped_train_X_bow.csv'
}

# feature vectors per word model
train_data = {}

for f in data_csv.keys():
    d = pd.read_csv('../data/' + data_csv[f])

    train_data[f] = d

y = pd.read_csv('../data/prepped_train_y.csv').as_matrix()

# remove the first column containing index numbers
y = np.delete(y, 0, 1)

OVR_ESTIMATORS = {
    "BR Random Forest": OneVsRestClassifier(RandomForestClassifier(n_estimators = 100)),
    "BR LinearSVC": OneVsRestClassifier(LinearSVC(random_state=1)),
    "BR Gaussian Naive Bayes": OneVsRestClassifier(GaussianNB()),
    "BR Bernoulli Naive Bayes": OneVsRestClassifier(BernoulliNB()),
}
BR_ESTIMATORS = {
    "BR Random Forest": BinaryRelevance(RandomForestClassifier(n_estimators = 100)),
    "BR LinearSVC": BinaryRelevance(LinearSVC(random_state=1)),
    "BR Gaussian Naive Bayes": BinaryRelevance(GaussianNB()),
    "BR Bernoulli Naive Bayes": BinaryRelevance(BernoulliNB()),
}
LP_ESTIMATORS = {
    "LP Random Forest": LabelPowerset(RandomForestClassifier(n_estimators = 100)),
    "LP LinearSVC": LabelPowerset(LinearSVC(random_state=1)),
    "LP Gaussian Naive Bayes": LabelPowerset(GaussianNB()),
    "LP Bernoulli Naive Bayes": LabelPowerset(BernoulliNB()),
}

# merge all dicts
ESTIMATORS = BR_ESTIMATORS.copy()
ESTIMATORS.update(LP_ESTIMATORS)

from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import clone as sk_clone
import time

def train(e, X, y):
    """
    Train all the estimators on the current dataset.
    The fit method should reset internals anyway.
    """
    e.fit(X, y)


def test(e, X, y):
    """calculating metrics based on the training set"""

    for metric in METRICS:
        cv = cross_validation.ShuffleSplit(len(y), random_state=0)
        scores = cross_validation.cross_val_score(e, X, y, cv=cv, scoring=metric)

        print "\t\tmean %s: %s" % (metric, scores.sum() / 10)

def run_est(X, y):
    """
    Train and test the estimators on the given dataset
    """
    tic = time.time()

    # all means of given METRICS
    means = []

    for e_name, e in ESTIMATORS.items():
        print "\t-> testing ", e_name

        ms = test(e, X, y)
        print "\t-> %ds elapsed for testing" % (time.time() - tic,)

        means.append(ms)

    return means

for k in train_data.keys():
    # convert pandas dataframe to np.array
    X = train_data[k].as_matrix()
    # remove continuous index numbers
    X = np.delete(X, 0, 1)

    assert(X.shape[0] == len(y))

    print "[#] Dataset: " + k
    run_est(X, y)

