# -*- coding: utf-8 -*-
"""
@author: MikeShuser
Module for often used data related functions

dependencies: 
    numpy >= 1.17
    pandas >= 0.23
"""

def crawl_directory(path: str, flag: str):
    '''
    Crawls through all folders in 'path', searching for 'flag' keyword
    Returns list of tuples with (folder_in_path, filename_with_flag)
    '''
    dir_list = []
    file_list = []
    for (_, dirname, files) in os.walk(path):
        dir_list.extend(dirname)
        file_list.extend([file for file in files if flag in file])        
    return list(zip(dir_list, file_list))