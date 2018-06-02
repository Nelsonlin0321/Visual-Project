# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:59:26 2018

@author: DELL
"""
import pandas as pd
import numpy as np
#multiclass
data_orig = pd.read_csv('winequality-red.csv')

data_X = data_orig.drop(['quality'],axis = 1)
data_y = data_orig['quality']

#将数据分类测试数据与训练数据

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_X,data_y,random_state = 0)
print('-'*10+'DecisionTreeRegressor'+'-'*10)
import scipy as sp
from sklearn.tree import DecisionTreeRegressor
#depth_list = np.linspace(1,200,100,dtype = 'int')
depth_list = np.arange(300)+1

train_accuracy = []
test_accuracy = []
detph_argument = 5
classifier_DTR=DecisionTreeRegressor(max_depth= detph_argument)
classifier_DTR.fit(X_train,y_train)
'''
test_dif = classifier_DTR.predict(X_test)-y_test
train_dif = classifier_DTR.predict(X_train)-y_train
test_err= sum((test_dif)**2)
train_err= sum((train_dif)**2)
'''
test_accur = (classifier_DTR.score(X_test,y_test))
train_accur= (classifier_DTR.score(X_train,y_train))
print('{:.4f}'.format(test_accur))
print('{:.4f}'.format(train_accur))

#def Decision_Tree_R(detph_argument = 2):
#  classifier_DTR=DecisionTreeRegressor(max_depth= detph_argument)
#  classifier_DTR.fit(X_train,y_train)
#  test_dif = classifier_DTR.predict(X_test)-y_test
#  train_dif = classifier_DTR.predict(X_train)-y_train
#  test_err= sp.sqrt(sp.mean(test_dif)**2)
#  train_err= test_err= sp.sqrt(sp.mean(train_dif)**2)
#  return test_err, train_err
#
#from concurrent.futures import ThreadPoolExecutor
#with ThreadPoolExecutor(max_workers = 20) as executor:
#  task_list = executor.map(Decision_Tree_R, depth_list) #一次返回的是 test train accuracy
#  
#  for accuracy in task_list:
#    print(accuracy[0])
#    print(accuracy[1])
#    print('---------------')
#    test_accuracy.append(accuracy[0])
#    train_accuracy.append(accuracy[1])
#    
#
#max_test_accuracy = np.argmax(test_accuracy)
#
##画图
#import matplotlib.pyplot as plt
#
#fig  = plt.figure(figsize = (10,8))
#ax = fig.add_subplot(1,1,1)
#ax.set_xlabel('Depth_Num')
#ax.set_ylabel('Accuracy')
#ax.plot(depth_list,train_accuracy,label = 'train accuracy')
#ax.plot(depth_list,test_accuracy,label = 'test accuracy')
#
#ax.plot(depth_list[max_test_accuracy],
#        test_accuracy[max_test_accuracy],
#        'o',markersize = 10, mew =2, 
#        fillstyle = 'none',c = 'r',
#        label = 'Max test accuracy ')
#
#ax.set_title('Decision_Tree_Regression-Accuracy')
#
## 设置数字标签  
##for a, b in zip(x1, y1):  
#ax.text(depth_list[max_test_accuracy],
#        test_accuracy[max_test_accuracy],
#        test_accuracy[max_test_accuracy], ha='center', va='bottom',fontsize = 20)
#    
## 设置直线
#ax.axvline(x = depth_list[max_test_accuracy],linestyle = '--')
#plt.legend(loc = 4)
#plt.show()