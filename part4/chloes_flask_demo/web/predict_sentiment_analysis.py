
import nltk
nltk.download('wordnet')

from nltk import data
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
import re
from nltk.util import everygrams
import pickle
from nltk.tokenize import word_tokenize

#from nltk import download
download('punkt',download_dir=data.path[0])
download('stopwords',download_dir=data.path[0])
download('wordnet',download_dir=data.path[0])
# #THEN remove the zip files!

data.path=['nltk_data']
stopwords_eng = stopwords.words("english")

lemmatizer = WordNetLemmatizer()

def extract_features(document):
    words = word_tokenize(document)
    lemmas = [str(lemmatizer.lemmatize(w)) for w in words if w not in stopwords_eng and w not in punctuation]
    document = " ".join(lemmas)
    document = document.lower()
    document = re.sub(r'[^a-zA-Z0-9\s]', ' ', document)
    words = [w for w in document.split(" ") if w!="" and w not in stopwords_eng and w not in punctuation]
    return [str('_'.join(ngram)) for ngram in list(everygrams(words, max_len=3))]

def bag_of_words(words):
    bag = {}
    for w in words:
        bag[w] = bag.get(w,0)+1
    return bag


import pickle
import sys

if not 'google.colab' in sys.modules:
    model_file = open('sa_classifier.pickle', 'rb')
    model = pickle.load(model_file)
    model_file.close()

from nltk.tokenize import word_tokenize

def get_sentiment(review):
    words = extract_features(review)
    words = bag_of_words(words)
    return model.classify(words)
