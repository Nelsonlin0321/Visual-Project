# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:05:14 2018

@author: DELL
"""

import pandas as pd
import numpy as np
import pydotplus
#multiclass
data_orig = pd.read_csv('winequality-red.csv')

data_X = data_orig.drop(['quality'],axis = 1)
data_y = data_orig['quality']

#将数据分类测试数据与训练数据

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_X,data_y,
                                                    random_state = 0,
                                                    test_size = 0.25)

print('-'*10+'DecisionTreeClassifier'+'-'*10)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


#depth_list = np.linspace(1,200,100,dtype = 'int')
depth_list = np.arange(300)+1

train_accuracy = []
test_accuracy = []


def Decision_Tree_C(detph_argument = 2):
  classifier_DT=DecisionTreeClassifier(max_depth= detph_argument)
  classifier_DT.fit(X_train,y_train)
  test_accur = (classifier_DT.score(X_test,y_test))
  train_accur= (classifier_DT.score(X_train,y_train))
  return test_accur, train_accur

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers = 20) as executor:
  task_list = executor.map(Decision_Tree_C, depth_list) #一次返回的是 test train accuracy
  
  for accuracy in task_list:
    print(accuracy[0])
    print(accuracy[1])
    print('---------------')
    test_accuracy.append(accuracy[0])
    train_accuracy.append(accuracy[1])
    

max_test_accuracy = np.argmax(test_accuracy)

#画图
import matplotlib.pyplot as plt

fig  = plt.figure(figsize = (10,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Depth_Num')
ax.set_ylabel('Accuracy')
ax.plot(depth_list,train_accuracy,label = 'train accuracy')
ax.plot(depth_list,test_accuracy,label = 'test accuracy')

ax.plot(depth_list[max_test_accuracy],
        test_accuracy[max_test_accuracy],
        'o',markersize = 10, mew =2, 
        fillstyle = 'none',c = 'r',
        label = 'Max test accuracy ')

ax.set_title('Decision_Tree_Classifier-Accuracy')

# 设置数字标签  
#for a, b in zip(x1, y1):  
ax.text(depth_list[max_test_accuracy],
        test_accuracy[max_test_accuracy],
        '{:.3f}'.format(test_accuracy[max_test_accuracy]), ha='center', va='bottom',fontsize = 20)
    
# 设置直线
ax.axvline(x = depth_list[max_test_accuracy],linestyle = '--')
plt.legend(loc = 4)
plt.show()


#决策树可视化
from sklearn.externals.six import StringIO
#获取 最优的 depth_num 的深度：

depth_num = depth_list[max_test_accuracy]
classifier_DT_Best=DecisionTreeClassifier(max_depth= depth_num)
classifier_DT_Best.fit(X_train,y_train)
feature_names = data_X.columns
#target_names = str(np.unique(data_y).tolist())
target_names = ['quality 3','quality 4', 'quality 5',
                'quality 6','quality 7','quality 8']

dot_data = StringIO()
tree.export_graphviz(classifier_DT_Best,out_file = dot_data,
                     filled=True,rounded=True,
                     special_characters=True,
                     feature_names=feature_names,
                      class_names=target_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Red_WineTree.pdf")
print('Visible tree plot saved as pdf.')






