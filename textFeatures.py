import nltk
import numpy as np
import simplifyBook

# returns avg sentence length, variation in sentence length, lexical diversity
def lexical_features(text):
    wordlist = simplifyBook.simplify(text)
    sentences = simplifyBook.sentencetokenize_text(text)
    sentence_length = np.array([len(simplifyBook.wordtokenize(s)) for s in sentences])
    return [sentence_length.mean(), sentence_length.std(), len(set(wordlist)) / float(len(wordlist))]


# returns proportion of overlapping vocabulary in 2 texts
def vocab_overlap(text1, text2):
    vocab1 = set(simplifyBook.simplify(text1))
    vocab2 = set(simplifyBook.simplify(text2))
    commonwords = list(vocab1 & vocab2)
    return [len(commonwords)/float(len(vocab1)), len(commonwords)/float(len(vocab2))]


# returns top n vocab used
def most_used_words(text, n):
    tokens = simplifyBook.simplify(text)
    fdist = nltk.FreqDist(tokens)
    return fdist.most_common(n)


def compare_features_same(raw,raw2):
    arr1 = lexical_features(raw)
    # arr1[0]-> avg sentence . arr1[1]-> std of sentence length, arr1[2]->lex richness
    arr2 = lexical_features(raw2)
    sentence_length_diff = abs(arr1[0] - arr2[0])
    std_sent_diff = abs(arr1[1] - arr2[1])
    richness_diff = abs(arr1[2] - arr2[2])
    return [1, sentence_length_diff, std_sent_diff, richness_diff]

def compare_features_diff(raw,raw2):
    arr1 = lexical_features(raw)
    # arr1[0]-> avg sentence . arr1[1]-> std of sentence length, arr1[2]->lex richness
    arr2 = lexical_features(raw2)
    sentence_length_diff = abs(arr1[0] - arr2[0])
    std_sent_diff = abs(arr1[1] - arr2[1])
    richness_diff = abs(arr1[2] - arr2[2])
    return [0, sentence_length_diff, std_sent_diff, richness_diff]