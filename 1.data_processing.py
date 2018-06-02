# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:42:00 2018

@author: DELL
"""

def data_processing(location = 'winequality-red.csv'):
  import pandas as pd
  dir_read = 'orig_data/'
  file_location = dir_read + location
  red_wine = open(file_location)
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
  dir_name = 'data/'
  df_red_wine.to_csv(dir_name+location,index = False)

if __name__ == '__main__':
  data_processing('winequality-red.csv')
  data_processing('winequality-white.csv')
  print('Done')


  