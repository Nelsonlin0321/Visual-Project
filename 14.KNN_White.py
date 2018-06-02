# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:00:47 2018

@author: DELL
"""

import pandas as pd
import numpy as np
#multiclass
data_orig = pd.read_csv('winequality-white.csv')

data_X = data_orig.drop(['quality'],axis = 1)
data_y = data_orig['quality']

#将数据分类测试数据与训练数据

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_X,data_y,random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
print('-'*40+'KNeighborsClassifier'+'-'*40)

neighbors_settings = range(1,31)
n_neighbors = 10


#利用测试数据对模型进行评估

train_accuracy = []
test_accuracy = []
# 对每一个n_neighbor 遍历构建模型
for neighbor in neighbors_settings:
  classifier=KNeighborsClassifier(n_neighbors  = neighbor)
  classifier.fit(X_train,y_train)
  test_accuracy.append(classifier.score(X_test,y_test))
  train_accuracy.append(classifier.score(X_train,y_train))

#找出最大的test_accuracy 的索引
max_test_accuracy = np.argmax(test_accuracy)

#画图
import matplotlib.pyplot as plt

fig  = plt.figure(figsize = (10,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('n_neighbors')
ax.set_ylabel('accuracy')
ax.plot(neighbors_settings,train_accuracy,label = 'train accuracy')
ax.plot(neighbors_settings,test_accuracy,label = 'test accuracy')

ax.plot(neighbors_settings[max_test_accuracy],
        test_accuracy[max_test_accuracy],
        'o',markersize = 10, mew =2, 
        fillstyle = 'none',c = 'r',
        label = 'Max test accuracy ')

ax.set_title('KNeighbors-Accuracy')

# 设置数字标签  
#for a, b in zip(x1, y1):  
ax.text(neighbors_settings[max_test_accuracy],
        test_accuracy[max_test_accuracy],
        '{:.4f}'.format(test_accuracy[max_test_accuracy]), ha='center', va='bottom',fontsize = 20)
    
# 设置直线
ax.axvline(x = neighbors_settings[max_test_accuracy],linestyle = '--')
plt.legend(loc = 4)
plt.show()
