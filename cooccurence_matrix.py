# -*- coding: utf-8 -*-
"""
Generate co-occurrence matrix from a text corpus.
Useful for some visualizations as well as network graph analysis. 

dependencies: 
    numpy >= 1.17
    pandas >= 0.23
"""

__author__ = "Mike Shuser"

from collections import Counter
import pandas as pd

def get_coocc_matrix(
    corpus: list, 
    window: int,
    vocab_subset: list = []
) -> pd.DataFrame:

    """
    Calculate the co-occurence matrix of a text corpus

    Parameters:
    corpus - expected to be a list of sentences
    window - size of context window to the left & right of each key word
    vocab_subset - optional subset of global corpus vocab to use as DataFrame 
        index. Handy for some downstream applications where you might want to ignore 
        some words, ie. stop words
    """

    joined = ' '.join(corpus)
    context_words = set(joined.split())

    if vocab_subset == []:
        vocab = context_words
    else:
        vocab = set(vocab_subset)

    co_occ = {key : Counter({val : 0 for val in context_words if val != key})
        for key in vocab}

    for doc in corpus:
        word_list = doc.split()
        for i in range(len(word_list)):
            if word_list[i] in vocab:
                if i < window:
                    #when context window is cutoff by start of sentence
                    c = Counter(word_list[0 : i + window + 1])
                    del c[word_list[i]]
                    co_occ[word_list[i]] = co_occ[word_list[i]] + c

                elif i > (len(word_list) - window + 1):
                    #when context window is cutoff by end of sentence
                    c = Counter(word_list[i - window:])
                    del c[word_list[i]]
                    co_occ[word_list[i]] = co_occ[word_list[i]] + c

                else:
                    #context window is not cutoff by start/end of sentence
                    c = Counter(word_list[i - window : i + window + 1])
                    del c[word_list[i]]
                    co_occ[word_list[i]] = co_occ[word_list[i]] + c

    co_occ = {word : dict(co_occ[word]) for word in vocab}
    
    return pd.DataFrame.from_dict(co_occ, orient='index')

