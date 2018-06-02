# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:03:20 2018

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

'''
print('-'*10+'Neural_Network'+'-'*10)
from sklearn.neural_network import MLPClassifier
classifier_NW = MLPClassifier(solver='adam', alpha=1e-5, activation='logistic',
                    hidden_layer_sizes=(5,3,2), random_state=1,max_iter=2000)
classifier_NW.fit(X_train,y_train)

print("Test set score:{:.2f}".format(classifier_NW.score(X_test,y_test)))
'''
from sklearn.neural_network import MLPClassifier

iter_list = np.linspace(5,600,60,dtype = 'int')

train_accuracy = []
test_accuracy = []

def Neural_Network(iter_argument = 200):
  classifier_NW = MLPClassifier(solver='adam', alpha=1e-5,
                                activation='logistic',
                                hidden_layer_sizes=(5,3,2), 
                                random_state=1,
                                max_iter=iter_argument)
  classifier_NW.fit(X_train,y_train)
  test_accur = (classifier_NW.score(X_test,y_test))
  train_accur= (classifier_NW.score(X_train,y_train))
  return test_accur, train_accur

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers = 20) as executor:
  task_list = executor.map(Neural_Network, iter_list) #一次返回的是 test train accuracy
  
  for accuracy in task_list:
    print(accuracy[0])
    print(accuracy[1])
    print('---------------')
    test_accuracy.append(accuracy[0])
    train_accuracy.append(accuracy[1])
    
'''
for iter_argument in iter_list:
  classifier_NW = MLPClassifier(solver='adam', alpha=1e-5, activation='logistic',
                    hidden_layer_sizes=(5,3,2), random_state=1,max_iter=iter_argument)
  classifier_NW.fit(X_train,y_train)
  test_accuracy.append(classifier_NW.score(X_test,y_test))
  train_accuracy.append(classifier_NW.score(X_train,y_train))

#找出最大的test_accuracy 的索引

'''
max_test_accuracy = np.argmax(test_accuracy)

#画图
import matplotlib.pyplot as plt

fig  = plt.figure(figsize = (10,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Iter_time')
ax.set_ylabel('accuracy')
ax.plot(iter_list,train_accuracy,label = 'train accuracy')
ax.plot(iter_list,test_accuracy,label = 'test accuracy')

ax.plot(iter_list[max_test_accuracy],
        test_accuracy[max_test_accuracy],
        'o',markersize = 10, mew =2, 
        fillstyle = 'none',c = 'r',
        label = 'Max test accuracy ')

ax.set_title('Neural_Network-Accuracy')

# 设置数字标签  
#for a, b in zip(x1, y1):  
ax.text(iter_list[max_test_accuracy],
        test_accuracy[max_test_accuracy],
        '{:.4f}'.format(test_accuracy[max_test_accuracy]), ha='center', va='bottom',fontsize = 20)
    
# 设置直线
ax.axvline(x = iter_list[max_test_accuracy],linestyle = '--')
plt.legend(loc = 4)
plt.show()




