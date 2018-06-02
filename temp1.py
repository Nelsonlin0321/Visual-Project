# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:38:10 2018

@author: DELL
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_path ='winequality-white.csv'
wine = pd.read_csv(data_path)
attributes_values = wine.drop(['quality'],axis = 1)
cov = np.corrcoef(attributes_values.T)
cov_abs = np.abs(cov)
#创建一个DataFram
df_cov_abs = pd.DataFrame(cov_abs,index = wine.columns[:-1])
df_cov_abs.columns = wine.columns[:-1]
'''
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
'''
