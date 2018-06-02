# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:08:56 2018

@author: DELL
"""
def attr_corr(data_path = 'winequality-red.csv'):
  import pandas as pd
  import matplotlib.pyplot as plt
  import numpy as np
  wine = pd.read_csv(data_path)
  attributes_values = wine.drop(['quality'],axis = 1)
  
  cov = np.corrcoef(attributes_values.T)
  cov_abs = np.abs(cov)
  
  fig, ax = plt.subplots(figsize=(40,40))
  img = ax.matshow(cov_abs,cmap='Reds')
  #plt.cm.winter
  image_name = data_path[12:-4]+' wine'
  plt.colorbar(img, ticks=[0,1],)
  plt.title(image_name,fontsize = 70)
  plt.xticks(np.arange(len(attributes_values.keys())), attributes_values.keys(),fontsize = 25)
  plt.yticks(np.arange(len(attributes_values.keys())), attributes_values.keys(),fontsize = 25)
  plt.savefig('images/'+image_name+'.png')
  plt.show()
  
  #输出 excel 文件 检验 
  atttribute_names = attributes_values.columns.values.tolist()
  df_attribute_corr = pd.DataFrame(cov_abs,index = atttribute_names)
  df_attribute_corr.columns = atttribute_names
  df_attribute_corr.to_csv('excel/'+data_path[12:-4]+'_attr_corr.csv',index = True)
  
attr_corr()
attr_corr('winequality-white.csv')