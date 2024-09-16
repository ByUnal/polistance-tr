import re
import string
from nltk.corpus import stopwords
from vnlp import Normalizer


def lower_case_func(text):
    return Normalizer.lower_case(text)


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords.words('turkish')])


def remove_punctuation(text):
    puncs = '’!"#$%&\'*+:;<=>?...@[\\]^_`{|}~“”'

    # Remove punctuation
    return text.translate(str.maketrans('', '', puncs))


def preprocess(text):
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = remove_punctuation(text)

    # text = remove_stopwords(text)

    # Remove digits
    text = re.sub(r'[0-9]{2}', '', text)
    remove_digits = str.maketrans('', '', string.digits)
    text = text.translate(remove_digits)

    text = re.sub(' +', ' ', text)  # remove extra whitespaces
    text = re.sub(r'([^\w\s])\1+', r'\1', text)
    text = re.sub(r'\s?([^\w\s])\s?', ' ', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(' +', ' ', text)
    return text.strip()
