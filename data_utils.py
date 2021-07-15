# -*- coding: utf-8 -*-
"""
@author: MikeShuser
Module for often used data related functions

dependencies: 
    numpy >= 1.17
    pandas >= 0.23
"""
import os
import hashlib
import re
from typing import Tuple, List
import pandas as pd
import numpy as np

class WeightVector():
    '''
    Object to apply weights to a dataframe.
    
    
    '''
    respid = 'record'
    
    def __init__(self, df, cellmap=None):
        self.df = df
        if cellmap != None:
            self.cells = list(cellmap.keys())
            self.tgt_dict = cellmap
                    
            #get the current cell distribution
            self.counts = pd.pivot_table(self.df, 
                                         values=self.respid, 
                                         index=self.cells,
                                         aggfunc='count')
            self.dist = self.counts/self.counts.sum()
            self.dist.columns=['dist']
        
            #create a targets table with the desired weighted distribution
            self.targets = pd.DataFrame(columns=self.cells, index=self.counts.index)
            for i, cell in enumerate(self.cells):
                tmp = '_ndx_' + cell
                self.targets[tmp] = self.targets.index.get_level_values(cell)
        
            for k, v in self.tgt_dict.items():
                for i, val in enumerate(v):
                    tmp = self.targets.index.get_level_values(k).unique()[i]
                    self.targets.loc[self.targets['_ndx_' + k]==tmp, k] = val
        
            for col in self.targets.columns:
                if '_ndx_' in col:
                    self.targets.drop(columns=col, inplace=True)
        
            #compute the interlocking weighted distribution
            self.dist['tgt_dist'] = 1
            for cell in self.cells:
                self.dist['tgt_dist'] = (self.dist['tgt_dist'] *
                         self.targets[cell])
            
            #compute the weights vector
            self.dist['weights'] = self.dist['tgt_dist'] / self.dist['dist']
        
        else:
            self.cells = None
            self.counts = None
            self.dist = None
            self.targets = None
        
    def applyWeights(self, questions):
        #wdf = self.df.select_dtypes(include=[np.number]).copy()
        if isinstance(questions, list):
            wdf = self.df[questions].copy()
        else:
            print('Question(s) must be in list form')
            return
                
        conditions=[]
        for w in self.dist.index.tolist():
            group = []
            for i, val in enumerate(w):
                group.append((str(self.cells[i]) + '==' + str(val)))
                
            conditions.append(" & ".join(group))
                
        for i, cond in enumerate(conditions):
            mask = self.df.eval(cond)
            wdf.loc[mask,:] = self.df.loc[mask, questions] * self.dist.iat[i,-1]
            
        return wdf
            
    
class SummaryTable():
    '''
    Base class to output standard summary tables.
    
    output_type:
        0 - all aggregates in one table
        1 - one aggregate per table
    '''
    
    def __init__(self, df, breaks, var):
        self.df = df
        self.breaks = breaks
        self.var = var

            
    def output(self, percent:bool, output_type:int):
        
        if percent:
            aggfuncs = ['count']
        else:
            aggfuncs = ['mean', 'count', 'std']
            
        if output_type == 0:
            return self.singleTab(aggfuncs, percent)
        
    def singleTab(self, aggfuncs, percent):
        tab = []
        first = True
        for br in self.breaks:
            if first:
                pvt = pd.pivot_table(self.df,
                                     index=br,
                                     values=self.var,
                                     aggfunc=aggfuncs,
                                     margins=True,
                                     margins_name="Total").reset_index()
                ndx = pvt.index.tolist()
                ndx.insert(0, ndx.pop())
                pvt = pvt.reindex(ndx)
                first=False
            else:
                pvt = pd.pivot_table(self.df,
                                     index=br,
                                     values=self.var,
                                     aggfunc=aggfuncs,
                                     margins=False).reset_index()
                
            pvt.insert(loc=0, column='break', value=br)
            pvt.rename(columns={br:'value'},inplace=True)
            tab.append(pvt)
        
        pvt = pd.concat(tab, axis=0, ignore_index=True)
        pvt.columns = pvt.columns.droplevel(1)
        pvt.loc[pvt['value']=='Total', 'break']='Total'
        pvt.index = [self.var] * len(pvt)
        #pvt.index = pvt['break']
        #pvt.index.name = self.var
        
        if percent:
            breaktotals = pvt.groupby('break')['count'].transform('sum')
            pvt['percent'] = pvt['count'] / breaktotals
            
        return pvt



def crawl_directory(path: str, flag: str) -> List[Tuple[str, str]]:
    """
    Crawls through all folders in 'path', searching for 'flag' keyword
    Returns list of tuples with (folder_in_path, filename_with_flag)
    """
    dir_list = []
    file_list = []
    for (_, dirname, files) in os.walk(path):
        dir_list.extend(dirname)
        file_list.extend([file for file in files if flag in file])        
    return list(zip(dir_list, file_list))


def hash_data_col(df: pd.DataFrame,
    hash_columns: List[str, int],
    missing_to_nan: bool = True) -> pd.DataFrame:
    """
    SHA256 encode a column of data

    Parameters:
    df - data source, as a pd.DataFrame
    has_columns - which columns to hash. Can be a list of names or indexes
    missing_to_nan - whether to convert rows to NaN, rather than hashing them
    """

    #SHA256 encodes an empty row as below
    missing = '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0'

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

        tgt[col] = src[col].apply(
            lambda x: hashlib.sha256(f"{x}".encode('utf-8')).hexdigest())

        if missing_to_nan:
            tgt[col].replace(missing, np.nan, inplace=True)

    return hashed_df

def flag_ip_addresses(ip_filter: List[str],
    ip_addresses: pd.Series) -> pd.Series:
    """
    Flag user IP addresses, usually a column in a dataframe, against a list
        of IP addresses to be blocked. Returns a boolean Series where True 
        values correspond to positive matches against ip_filter.
        Currently, this implementation only accounts for IPv4.
    
    Parameters:
    ip_filter - list of IPs to flag/remove. IPs including netmask is expected
    ip_addresses - Series of source IPs
    """
    import ipaddress

    ips_to_flag = [ipaddress.IPv4Network(x) for x in ip_filter]
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


def censor_profanity():
    eng_pf = pd.read_excel("_admin_files/profanity_filter.xlsx",
                           sheet_name="english").english

    for i in oe_df.index:
        response = str(oe_df.at[i,'OpenEnd'])
        change_flag = False

        for word in eng_pf:
            pattern = re.compile(r'\b({0})\b'.format(word), 
                                 flags=re.IGNORECASE) 
            if bool(re.search(pattern, response)):
                change_flag = True
                repl = '*' * len(word)
                #print(i, word)
                #print("OLD: " + response)
                #print("NEW: " + re.sub(pattern, repl, response))
                #print()
                
                response = re.sub(pattern, repl, response)

        
        if change_flag:
            oe_df.at[i,'OpenEnd'] = response

def search_keywords():
    search_cols = ['B4B-WM']
    keywords = pd.read_excel("C:\\DataPrep\\_admin_files\\keyword_search.xlsx",
                             sheet_name="mask").keywords

    res = pd.Series(index=df.index, dtype="bool")
    for col in search_cols:
        for word in keywords:
            res = df[col].str.contains(fr"\b{word}\b", case=False, na=False, regex=True) | res
        
    res.astype(int).to_excel("C:\\RMG\\3- Store track\\ad hoc\\res.xlsx")
