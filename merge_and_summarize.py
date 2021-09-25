# -*- coding: utf-8 -*-
"""
Merge multiple data files and summarize in excel 
"""

__author__ = "Mike Shuser"

import glob
import pandas as pd
import numpy as np
import xlwings as xl

def compute_weighted_scores(df):
    drivers = [
        'instock',
        'fast',
        'friendly',
        'clean',
        'price',
        'quality',
    ]
    all_data = df[drivers].copy()
    all_data['friendly'].replace(99, np.nan, inplace=True)
    
    df_wFriendly = all_data.loc[all_data['friendly'].notnull(), :].copy()
    df_noFriendly = all_data.loc[all_data['friendly'].isnull(), :].copy()
    df_noFriendly.friendly.replace(np.nan, 0, inplace=True)
    
    dsat_wFriendly = [0.18, 0.21, 0.21, 0.13, 0.16, 0.11]
    cf4_wFriendly = [0.21, 0.24, 0.37, 0.18, 0, 0]
    weights_wFriendly = np.array([dsat_wFriendly, cf4_wFriendly]).T
    
    dsat_noFriendly = [0.2278481, 0.2658228, 0, 0.1645570, 0.2025316, 0.1392405]
    cf4_noFriendly = [0.3333333, 0.3809524, 0, 0.2857143, 0, 0]
    weights_noFriendly = np.array([dsat_noFriendly, cf4_noFriendly]).T
    
    scores_wFriendly = df_wFriendly.to_numpy() @ weights_wFriendly
    scores_noFriendly = df_noFriendly.to_numpy() @ weights_noFriendly

    scores = np.vstack([scores_wFriendly, scores_noFriendly])
    ndx = np.hstack([df_wFriendly.index, df_noFriendly.index])
    
    final = pd.DataFrame(
        index=ndx, 
        data=scores,
        columns=['derived_sat', 'cf4']
    )
    
    return final

if __name__ == '__main__':

    path = "C:/CSAT_data/"

    column_layout = pd.read_excel(f"{path}/layout.xlsx")['var'].tolist()
    meta = glob.glob(f'{path}/*Alignment*.csv')[0]
    meta = pd.read_csv(meta, index_col=0)
    meta_cols = ['Province', 'Region', 'Market']

    fids = list(range(1,12))
    merged_col_ndx = ['derived_sat', 'cf4'] + column_layout
    merged = pd.DataFrame(columns=merged_col_ndx)

    #merge single files
    for fid in fids:
        print(fid)
        query = glob.glob(f"{path}/raw_monthly/CSAT_m{fid}_*.csv")
        assert len(query) == 1, "multiple files found, review source directory"
        
        df = pd.read_csv(query[0])
        
        derived = compute_weighted_scores(df)
        df = df.merge(derived, left_index=True, right_index=True)
        
        df = df[merged_col_ndx]
        merged = pd.concat([merged, df], ignore_index=True, sort=False)

    means = merged.groupby('store_num').mean()
    counts = merged.groupby('store_num').count()

    for col in meta_cols:
        means[col] = means.index.map(meta[col])
        counts[col] = counts.index.map(meta[col])

    wb = xl.Book()
    ws = wb.sheets[0]
    ws.range(1, 1).options(index=True).value = means
    ws.range(1, len(merged_col_ndx) + 3).options(index=True).value = counts


