# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def distribution_corr(data_path = 'winequality-red.csv',
                    attribute = 'fixed acidity'):    
  wine = pd.read_csv(data_path)
  quality = wine.quality
  attribute_values = wine[attribute]
  corr = attribute_values.corr(quality)
  unique_quality = np.unique(quality)
  color_list = ['red','green','blue','orange',
                'pink','gray','purple','yellow','black']
  color_num = 0
  plt.figure(figsize=(14,8))
  for quality in unique_quality:
    
    red_wine_quality = wine[wine.quality == quality]
    #遍历每一个quality 的 acidity
    attribute_data = red_wine_quality[attribute]
    sns.distplot(attribute_data,rug=True, 
                 hist=True,color = color_list[color_num],
                 label = 'quality {}'.format(quality))
    color_num +=1
    
    plt.legend(loc = 2)
  
  plt.ylabel('Density',fontsize = 20)
  plt.xlabel(attribute,fontsize = 20)
  plt.title(attribute +' Distribution'+ ' Correlation :{:.3f}'.format(corr),
            fontsize=20)
  attribute = attribute.replace('\n',' ')
  plt.savefig('images/'+data_path[:-4]+'/'+attribute+'.png')

red_wine = pd.read_csv('winequality-red.csv')
attribute_names_red = red_wine.columns.values.tolist()[:-1]

for attribute in attribute_names_red:
  distribution_corr('winequality-red.csv',attribute)


white_wine = pd.read_csv('winequality-white.csv')
attribute_names_white = white_wine.columns.values.tolist()[:-1]
for attribute in attribute_names_white:
  distribution_corr('winequality-white.csv',attribute)

print('Completed')

#testing
'''
red_wine = pd.read_csv('winequality-red.csv')
attribute_names_red = red_wine.columns.values.tolist()[:-1]
distribution_corr('winequality-red.csv',attribute_names_red[0])
'''