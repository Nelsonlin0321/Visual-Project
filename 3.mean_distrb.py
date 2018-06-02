# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:07:23 2018

@author: DELL

"""

def find_corr(data_path ='winequality-red.csv',attribute = 'fixed acidity'):
  import pandas as pd
  import numpy as np
  import matplotlib.pylab as plt
  #读取数据
  red_wine = pd.read_csv(data_path)
  
  #fixed_acidity 相关性分析
  #fa_qulit = red_wine[['fixed acidity','quality']]
  
  ##fixed_acidity 和quality的相关系数
  fixed_acidity = red_wine[attribute]
  quality = red_wine['quality'] 
  corr = fixed_acidity.corr(quality)
  corr = '{:.2f}'.format(corr)
  
  fa_qulit = red_wine.pivot_table(index = 'quality', values = attribute, aggfunc=np.mean)
  fa_qulit = fa_qulit.reset_index() 
  
  #index = np.arange(fa_qulit['fixed acidity'].size)
  plt.figure(figsize=(12,6))
  plt.bar(fa_qulit['quality'],fa_qulit[attribute]) 
  plt.ylabel('Mean '+ attribute,fontsize = 12)
  plt.xlabel('Quality',fontsize = 12)
  plt.title('Quanlity-{}:Corr '.format(attribute)+ corr,fontsize = 16)
  attribute = attribute.replace('\n',' ')
  plt.savefig('images/'+data_path[:-4]+'/'+attribute +'.jpg')
  plt.show()
  

import pandas as pd
red_wine = pd.read_csv('winequality-red.csv')
attribute_names_red = red_wine.columns.values.tolist()[:-1]

for attribute in attribute_names_red:
  find_corr('winequality-red.csv',attribute)

white_wine = pd.read_csv('winequality-white.csv')
attribute_names_white = white_wine.columns.values.tolist()[:-1]
for attribute in attribute_names_white:
  find_corr('winequality-white.csv',attribute)

  


