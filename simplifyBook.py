import nltk
import numpy as np


def wordtokenize(text):
    text = text.lower()
    # create an array of tokens
    tokens = nltk.word_tokenize(text)
    # remove punctuations
    tokens = [word for word in tokens if word[0].isalpha()]
    return sorted(tokens)


def remove_stopwords(tokens):
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stopwords]
    return tokens


# returns tokenized text in an array, without stopwords
def simplify(text):
    return remove_stopwords(wordtokenize(text))


# returns sentences in text
def sentencetokenize_text(text):
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return sentence_tokenizer.tokenize(text)

