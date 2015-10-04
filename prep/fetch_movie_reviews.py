import pandas as pd
import re
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import word2vec


# punkt tokenizer to split sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

train = pd.read_csv("../data/descr_and_fid.csv", delimiter=",")

def prep_sent(sent, remove_stops=False):
    # Step 1: remove non-letters
    p = re.sub("^[a-zA-Z]", " ", sent)

    words = p.lower().split()

    if remove_stops:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return (words)

def plot_to_sents(plot, tokenizer, remove_stops=False):
    sents = []

    raw = tokenizer.tokenize(plot.strip())

    for r in raw:
        if len(r) > 0:
            sents.append(prep_sent(r, remove_stops))

    return sents


descs = []

print "Preparing descriptions"
for desc in train["description_en"]:
    if type(desc) == str:
        descs += plot_to_sents(desc.decode("utf-8"), tokenizer)

print "Learning word2vec"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 2       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

model = word2vec.Word2Vec(descs, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)
