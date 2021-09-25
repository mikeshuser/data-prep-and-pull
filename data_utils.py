# -*- coding: utf-8 -*-
"""
Module for often used data related functions

dependencies: 
    numpy >= 1.17
    pandas >= 0.23
"""

__author__ = "Mike Shuser"

import os
import hashlib
import re
from typing import Tuple, List, Union
import pandas as pd
import numpy as np

def crawl_directory(path: str, flag: str) -> List[Tuple[str, str]]:

    """
    Crawls through all folders in 'path', searching for 'flag' keyword
    Returns list of tuples with (folder_in_path, filename_with_flag)
    """

    res = []
    for (parent, _, files) in os.walk(path):
        for file in files:
            if flag in file:
                res.append((parent, file))
        
    return res

def hash_data_col(
    df: pd.DataFrame,
    hash_columns: List[Union[str, int]],
    missing_to_nan: bool = True,
) -> pd.DataFrame:
    
    """
    SHA256 encode a column of data

    Parameters:
    df - data source, as a pd.DataFrame
    has_columns - which columns to hash. Can be a list of names or indexes
    missing_to_nan - whether to convert rows to NaN, rather than hashing them
    """

    import hashlib
    assert isinstance(hash_columns, list), "hash_columns needs to be a list"

    hashed_df = df.copy()
    for col in hash_columns:

        if isinstance(col, int):
            tgt = hashed_df.iloc
            src = df.iloc
        elif isinstance(col, str):
            
            tgt = hashed_df.loc
            src = df.loc
        else:
            return "Improper type found in hash_columns"

        tgt[:, col] = src[:, col].apply(
            lambda x: hashlib.sha256(f"{x}".encode('utf-8')).hexdigest())

        if missing_to_nan:
            missing = src[:, col].isnull()
            tgt[:, col].mask(missing, inplace=True)

    return hashed_df

def flag_ip_addresses(
    ip_filter: List[str],
    ip_addresses: pd.Series,
) -> pd.Series:
    """
    Flag user IP addresses, usually a column in a dataframe, against a list
        of IP addresses to be blocked. Returns a boolean Series where True 
        values correspond to positive matches against ip_filter.
        Currently, this implementation only accounts for IPv4.
    
    Parameters:
    ip_filter - list of IPs to flag/remove. IPs with subnet mask expected
    ip_addresses - Series of source IPs
    """
    import ipaddress

    ips_to_flag = [ipaddress.IPv4Network(x, False) for x in ip_filter]
    user_ips = pd.Series(index=ip_addresses.index, dtype=str)

    #convert str IPs to IP objects. Invalid formats turned to 255.255.255.255
    for i, v in enumerate(ip_addresses):
        try:
            user_ips.iat[i] = ipaddress.IPv4Network(v)
        except:
            print(f"Invalid IPv4 address at {i}: {v}")
            user_ips.iat[i] = ipaddress.IPv4Network('255.255.255.255')
    
    flagged = pd.Series(index=ip_addresses.index, data=False)
    for ip in ips_to_flag:
        flagged = user_ips.apply(lambda y: ip.supernet_of(y)) | flagged    

    #TODO: add IPv6 implementation

    return flagged

def censor_profanity(
    data_col: pd.Series, 
    swear_words: List[str],
) -> pd.Series:
    
    """
    Pattern matching algo that censors profanity in a text response.
    
    Example input:
    'Customer service at this store was absolute shit.' 
        becomes ->
    'Customer service at this store was absolute ****.' 
    """
    
    res_col = data_col.copy()
    for i in data_col.index:
        response = str(data_col.at[i])
        change_flag = False

        for word in swear_words:
            pattern = re.compile(r'({0})'.format(word), 
                                 flags=re.IGNORECASE) 
            if bool(re.search(pattern, response)):
                change_flag = True
                repl = '*' * len(word)
                response = re.sub(pattern, repl, response)
        
        if change_flag:
            res_col.at[i] = response

    return res_col

def search_keywords(
    data_src: pd.DataFrame,
    search_cols: List[Union[str, int]],
    keywords: List[str],
) -> pd.Series:

    """
    Keyword search an arbitrary number of columns in a dataframe.
    search_col can be integer index or column name.
    Returns a boolean array where any row containing a 
        text response that matched a keyword is True. 
    """

    res = pd.Series(index=data_src.index, dtype="bool")
    for col in search_cols:
        
        if isinstance(col, int):
            src = data_src.iloc[:, col]
        elif isinstance(col, str):
            src = data_src.loc[:, col]
        else:
            return "Improper type found in search_cols"

        for word in keywords:
            res = src.str.contains(fr"\b{word}\b", 
                case=False, na=False, regex=True) | res
    
    return res

def compute_weighted_avg(
    df: pd.DataFrame,
    weights: np.ndarray,
    subset: List[str] = [],
) -> pd.DataFrame:

    """
    Compute a weighted avg of a dataframe columns/subset of columns.
    Weights should be a numpy array where each row sums to 1, and column-wise 
        the weights represent the column list/subset.
    """

    if len(weights.shape) == 1:
        weights = np.expand_dims(weights, 0)

    assert (np.sum(weights, axis=1) == 1).all(),  \
        "weights should sum to 1 for each row"

    if subset == []:
        unweighted = df.to_numpy(copy=True)
    else:
        unweighted = df.loc[:, subset].to_numpy(copy=True) 

    assert unweighted.shape[1] == weights.shape[1], \
        "number of columns between weights and data matrix do not match"

    unweighted = np.nan_to_num(unweighted)
    weighted = unweighted @ weights.T
    
    return pd.DataFrame(index=df.index, data=weighted)