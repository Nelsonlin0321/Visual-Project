# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:02:32 2018

@author: DELL
"""

import pandas as pd
import matplotlib.pyplot as plt
def scatter_plot(location = 'winequality-red.csv'):
  wine = pd.read_csv(location)
  quality = wine['quality'] -3
  attributes_values = wine.drop(['quality'],axis = 1)
  pd.plotting.scatter_matrix(attributes_values,figsize = (40,40),
                             diagonal = 'hist',marker = 'o',
                             c = quality, s= 20)
  plt.title(location[:-4],fontsize = 30)
  plt.show()
  
scatter_plot()

scatter_plot('winequality-white.csv')
'''
#计算相关系数
corr_dic = {}
attribute_names  = attributes_values.columns.values.tolist()

for attribute1 in attribute_names:#获取一个attribute的名字
  print(attribute1)
  print('___'*20)
  corr_list = [] #储存他对其他attribute correlation
  attribute_values1 = attributes_values[attribute1] #获取这个 attribute的values
  for attribute2 in attribute_names:
    attribute_values2 = attributes_values[attribute2]
    corr = attribute_values2.corr(attribute_values1)
    print(attribute2, end = ' ')
    print(corr)
    corr_list.append(corr)
  corr_dic[attribute1] = corr_list

df_corr = pd.DataFrame(corr_dic,index = attribute_names)
#输出coor 表
df_corr.to_csv('excel/attribute_corr.csv',index = True)

cov = np.corrcoef(attributes_values.T)
cov_abs = np.abs(cov)

fig, ax = plt.subplots(figsize=(40,40))
img = ax.matshow(cov_abs,cmap='Reds')
#plt.cm.winter
plt.colorbar(img, ticks=[0,1],)
plt.xticks(np.arange(len(attributes_values.keys())), attributes_values.keys(),fontsize = 25)
plt.yticks(np.arange(len(attributes_values.keys())), attributes_values.keys(),fontsize = 25)
plt.show()

#输出 excel 文件 检验 
atttribute_names = attributes_values.columns.values.tolist()
df_attribute_corr = pd.DataFrame(cov_abs,index = atttribute_names)
df_attribute_corr.columns = atttribute_names
df_attribute_corr.to_csv('excel/attribute_corr.csv',index = True)
'''





    
    