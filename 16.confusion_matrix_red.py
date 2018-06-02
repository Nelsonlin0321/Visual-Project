# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:21:33 2018

@author: DELL
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(15,15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#--------------------------------------------------------------
data_orig = pd.read_csv('winequality-red.csv')

data_X = data_orig.drop(['quality'],axis = 1)
data_y = data_orig['quality']

#将数据分类测试数据与训练数据
X_train, X_test, y_train, y_test = train_test_split(data_X,data_y,
                                                    random_state = 0,
                                                    test_size = 0.25)
#---------------------------------------------------------------------
print('-'*10+'DecisionTreeClassifier'+'-'*10)
from sklearn.tree import DecisionTreeClassifier


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
#-----------------------------------------------------------------------
depth_num = depth_list[max_test_accuracy]
classifier_DT_Best=DecisionTreeClassifier(max_depth= depth_num)
classifier_DT_Best.fit(X_train,y_train)
y_pred = classifier_DT_Best.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = np.unique(data_y)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()