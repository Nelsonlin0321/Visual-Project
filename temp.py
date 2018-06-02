# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:22:56 2018

@author: DELL
"""

import pandas as pd
red_wine_location = 'winequality-red.csv'

red_wine = open(red_wine_location)
list = []
for line in red_wine:
  line = line.strip()
  word_list = line.split(';')
  list.append(word_list)

attributes = list[0]
values = list[1:]

filter_attributes = []

for attribute in attributes:
    attribute= attribute[1:-1]
    filter_attributes.append(attribute)


df_red_wine = pd.DataFrame(values)
df_red_wine.columns = filter_attributes
df_red_wine.to_csv('red_wine.csv',index = False)