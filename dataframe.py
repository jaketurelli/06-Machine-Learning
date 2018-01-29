# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:30:36 2018

@author: Jake
"""

import pandas as pd
import numpy as np
exam_data = [{'name':'Anastasia', 'score':12.5}, {'name':'Dima','score':9}, {'name':'Katherine','score':16.5}]
df = pd.DataFrame(exam_data)
for index, row in df.iterrows():
    print(row['name'], row['score'])