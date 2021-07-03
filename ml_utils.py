# -*- coding: utf-8 -*-
"""
@author: MikeShuser
Module housing some useful functions for machine learning

dependencies: 
    numpy >= 1.17
    pandas >= 0.23
    sklearn >= 0.21.2
"""
import random
import os
import numpy as np
import pandas as pd

def get_sample_weights(labels: pd.Series, ndx_subset: list = []):
    """
    Return a Series of weights that results in an equal-weighted 
        class distribution
    ndx_subset(optional): pass an index list to filter the label series 
        into some desired subset 
    """
    #filter the label series if applicable
    if ndx_subset != []:
        y = labels.loc[ndx_subset].copy()
    else:
        y = labels.copy()
        
    klasses = y.unique().tolist()
    weights = {}
    for klass in klasses:
        weights[klass] = (len(y) / (y==klass).sum()) / len(klasses)

    return y.replace(weights)

def partition_class(y_col: pd.Series, 
    ratio: list = [80, 10, 10],
    remove_nan: bool = False):
    """
    Split a target or output series into train/val/test partitions, 
        following the same distribution as 'ratio' arg
    """
    assert sum(ratio) == 100, "ratio needs to sum to 100"

    labels = pd.Series(index=y_col.index, dtype=str)
    if remove_nan:
        klass_col = y_col.loc[y_col.notnull()].copy()
    else:
        klass_col = y_col.fillna(value=0) #missing treated as negative class

    klass_col = klass_col.sample(frac=1.0, replace=False) #shuffle labels

    klasses = klass_col.unique()
    segmented = [klass_col[klass_col==klass] for klass in klasses]

    #begin partitioning
    train_pieces = []
    val_pieces = []
    test_pieces = []
    splitA = ratio[0]/100
    splitB = sum(ratio[:2])/100
    for piece in segmented:
        p_train, p_val, p_test = np.split(piece,
            [int(splitA * len(piece)), int(splitB * len(piece))])

        train_pieces.append(p_train)
        val_pieces.append(p_val)
        test_pieces.append(p_test)

    train = pd.concat(train_pieces)
    val = pd.concat(val_pieces)
    test = pd.concat(test_pieces)

    for i in labels.index:
        if i in train.index:
            labels.at[i] = 'train'
        elif i in val.index:
            labels.at[i] = 'val'
        elif i in test.index:
            labels.at[i] = 'test'

    return labels

def interactions_ignore_nan(df: pd.DataFrame, level=2):
    """
    Expand a feature matrix with multiplicative interactions
    Feature combinations can be 2(default) or 3 variables
    Missing values are ignored(replaced with 1)
    
    caution: be careful on which dataframes you run this function.
        It can explode your feature dimensions and/or cause memory overflow
    """
    from itertools import combinations
    
    columns = df.columns.tolist()
    if level==2:
        groups = list(combinations(columns,2))
    elif level==3:
        groups = list(combinations(columns,2))
        groups.extend(list(combinations(columns,3)))
        
    tmp_df = df.copy()
    for group in groups:
        base = tmp_df.loc[:,group[0]]
        for prod in group[1:]:
            base = base.multiply(tmp_df.loc[:,prod], fill_value=1)
        tmp_df[group] = base
 
    return tmp_df

def find_random_seed(X, Y, split: float = 0.2, attempts: int = 10):
    """
    Search for a seed that produces a desired class distribution 
        from train_test_split

    assumes Y == {0,1} 
    """
    from sklearn.model_selection import train_test_split
    
    for i in range(10):
        seed = random.randint(1,100)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                            test_size=split, 
                                                            random_state=seed)
        
        #ratio of 1 means train/test have equal distribution
        #ratio > 1 means positive class in train > test, and vice versa  
        ratio = ((Y_train==1).sum() / (Y_train==0).sum()) / \
            ((Y_test==1).sum() / (Y_test==0).sum())
            
        print(f"iteration {i}, ratio {ratio}")
        print(f"produced by seed {seed}")
        ans = input('Type "keep" to end loop: ')
        if ans == 'keep':
            break
    return seed

def import_embeddings(filepath: str):
    """
    Import word embeddings txt file (space separated) into a dict
    Used for word2vec, glove embeddings 
    """
    vecs = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            emb = np.array([float(val) for val in split_line[1:]])
            vecs[word] = emb
    return vecs
    
def shuffle_in_unison(n_arraylikes: list):
    """
    Shuffle multiple arrays of equal length in the same order
    """
    array_len = len(n_arraylikes[0])
    for arr in n_arraylikes:
        assert len(arr) == array_len
    p = np.random.permutation(array_len)
    shuffled = []

    for i in range(len(n_arraylikes)):
        if isinstance(n_arraylikes[i], np.ndarray):
            shuffled.append(n_arraylikes[i][p])
        elif isinstance(n_arraylikes[i], (pd.Series, pd.DataFrame)):
            shuffled.append(n_arraylikes[i].loc[p])
    
    return shuffled