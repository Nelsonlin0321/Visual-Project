# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 22:11:25 2018

@author: DELL
"""

import pandas as pd
import numpy as np

import seaborn as sns

#data_path = 'winequality-white.csv'
##创建字典 储存 attribute 和 corr 
#wine = pd.read_csv(data_path)
#quality = wine.quality
## 所有的attributes
#attributes = wine.columns.values.tolist()[:-1]
#corr_list = []
#for attribute in attributes:
#  #获取数据
#  attribute_values = wine[attribute]
#  corr = attribute_values.corr(quality)
#  corr_list.append(corr)
#  #传到dataFram中：
#df_attribute_corr = pd.DataFrame({'corr':corr_list},index = attributes)
#df_attribute_corr = df_attribute_corr.reset_index()
#df_attribute_corr.columns = ['attributes','corr']
#df_attribute_corr.sort_values('corr',inplace = True, ascending = True)
#
#index = range(df_attribute_corr.attributes.size)
#attributes_names = df_attribute_corr['attributes'].values.tolist()
#corr = df_attribute_corr['corr'].values.tolist()
#attribute_names = df_attribute_corr.attributes
#
#import matplotlib.pylab as plt
#plt.figure(figsize=(18,8))
#plt.title('Attribute_Correlation',fontsize = 18)
#plt.ylabel('Correlation',fontsize = 12)
#plt.barh(index,corr,tick_label = attribute_names,fc = 'y')
#for y, x in zip(index,corr):
#  plt.text(x,y, '%.3f' %x,
#           ha = 'center', va = 'center',fontsize = 11)

def correlation_chart(data_path = 'winequality-white.csv'):
  wine = pd.read_csv(data_path)
  quality = wine.quality
  # 所有的attributes
  attributes = wine.columns.values.tolist()[:-1]
  corr_list = []
  for attribute in attributes:
    #获取数据
    attribute_values = wine[attribute]
    corr = attribute_values.corr(quality)
    corr_list.append(corr)
    #传到dataFram中：
  df_attribute_corr = pd.DataFrame({'corr':corr_list},index = attributes)
  df_attribute_corr = df_attribute_corr.reset_index()
  
  df_attribute_corr.columns = ['attributes','corr']
  df_attribute_corr['corr'] = np.abs(df_attribute_corr['corr'])
  df_attribute_corr.sort_values('corr',inplace = True, ascending = True)
  df_attribute_corr.to_csv('excel/'+data_path[4:-4]+'_abs_corr.csv',index = False)
  
  index = range(df_attribute_corr.attributes.size)
  attribute_names = df_attribute_corr['attributes'].values.tolist()
  corr = df_attribute_corr['corr'].values.tolist()
  attribute_names = df_attribute_corr.attributes
  
  import matplotlib.pylab as plt
  plt.figure(figsize=(18,8))
  plt.title(data_path[:-4]+' :Abs_Correlation',fontsize = 18)
  plt.ylabel('Correlation',fontsize = 12)
  plt.barh(index,corr,tick_label = attribute_names,fc = 'y')
  for y, x in zip(index,corr):
    plt.text(x,y, '%.3f' %x,
             ha = 'center', va = 'center',fontsize = 11)

correlation_chart('winequality-red.csv')
correlation_chart()


  