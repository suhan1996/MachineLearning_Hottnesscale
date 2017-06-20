#!/usr/bin/env python2
# -*- coding: utf-8 -*-
API_KEY = 'a038c1cdf6f88543dea4dbd86530e67d'
API_SECRET = '8e7GMEPtWe3P2sG4tKwze6hWzyVydEp5'
import json
import csv
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pprint import pformat
import random
from sklearn import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA
from sklearn.externals import joblib
SAMPLE = 2500
def print_result(hint, result):
    def encode(obj):
        if type(obj) is unicode:
            return obj.encode('utf-8')
        if type(obj) is dict:
            return {encode(k): encode(v) for (k, v) in obj.iteritems()}
        if type(obj) is list:
            return [encode(i) for i in obj]
        return obj
    print hint
    result = encode(result)
    print '\n'.join(['  ' + i for i in pformat(result, width = 75).split('\n')])

def landmark():
    from facepp import *
    api = API(API_KEY, API_SECRET)
    landmark = {}
    invalid = []
    global SAMPLE
    for i in range(1,SAMPLE+1): #training + testing
        try:
            face = api.detection.detect(img = File('p ('+ str(i)+').jpg'))
            face_id = face['face'][0]['face_id']
            print "uploading face " + str(i)
            result = api.detection.landmark(face_id = face_id,type = '25p')
            landmark[i] = result
        except:
            invalid.append(i)


    f1 = open("landmark_2500.txt","wb")
    pickle.dump(landmark, f1)
    f1.close()

    f2 = open("invalid_2500.txt","wb")
    pickle.dump(invalid, f2)
    f2.close()

    return landmark,invalid

def getTraining(landmark,invalid): #nose_tip as reference
    global SAMPLE
    landmarkList = [u'mouth_upper_lip_top',u'right_eye_top',u'left_eye_bottom',u'mouth_lower_lip_bottom',
    u'right_eyebrow_left_corner',u'right_eye_bottom',u'right_eye_pupil',u'mouth_left_corner',u'left_eyebrow_left_corner',\
    u'nose_right',u'nose_tip',u'right_eye_right_corner',u'left_eye_left_corner',\
    u'mouth_right_corner',u'left_eyebrow_right_corner',u'right_eye_center',\
    u'nose_left',u'left_eye_center',u'right_eye_left_corner',u'mouth_lower_lip_top',\
    u'left_eye_pupil',u'right_eyebrow_right_corner',u'left_eye_top',u'left_eye_right_corner',
    u'mouth_upper_lip_bottom']
    #print landmark
    
    y = []
    reader = csv.reader(file('target.csv','rb'))
    i = 1
    count = 0
    for line in reader:
        if i not in invalid:
            y.append(line[1])
        i += 1
        count += 1
        if count == SAMPLE:
            break

    x = []
    count = 0
    for person in landmark:
        x.append([])
        for i in landmarkList:
            x[count].append(landmark[person][u'result'][0][u'landmark'][i][u'x']-landmark[person][u'result'][0][u'landmark'][u'nose_tip'][u'x'])
            x[count].append(landmark[person][u'result'][0][u'landmark'][i][u'y']-landmark[person][u'result'][0][u'landmark'][u'nose_tip'][u'y'])
        count += 1

    #a = clf.predict(X_test)
    #print a
    
    return x,y
    #----------- aggregation ---------------

def boost(x,y):
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
     "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
    classifiers = [
        KNeighborsClassifier(10),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=1, C=5),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA()]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        joblib.dump(clf, 'classifier/' + name + '.pkl')
        pre = clf.predict(X_test)
        zeroCount = 0
        matchCount = 0
        for i in range(len(y_test)):
            if y_test[i] == '0':
                zeroCount += 1
                if pre[i] == '0':
                    matchCount += 1
        print matchCount*1.0/zeroCount
  

        print "classifier:  "+ name + "   score:  "+ str(score)

def parameterRBF(x,y):
    gammaList = [0.1,1,10,100]
    CList = [0.1,0.2,0.5,1,5,10]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)
    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=.5)
    maxScore = 0
    for i in gammaList:
        for j in CList:
            clf = SVC(gamma=i, C=j)
            clf.fit(X_train, y_train)
            score = clf.score(X_cv, y_cv)
            print "(gamma,C) =   "+ str((i,j)) + "   score:  "+ str(score)
            if score > maxScore:
                maxScore = score
                maxGamma = i
                maxC = j
    score = clf.score(X_test, y_test)
    
    print "(gamma,C) =   "+ str((maxGamma,maxC)) + "   test_score:  "+ str(score)
    #   gamma = 1 C = 5

def parameterRF(x,y):
    estimatorsList = [5,10]
    featuresList = ["auto","sqrt","log2"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)
    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=.2)
    maxScore = 0
    for i in estimatorsList:
        for j in featuresList:
            clf = RandomForestClassifier(max_depth=5, n_estimators=i, max_features=j)
            clf.fit(X_train, y_train)
            score = clf.score(X_cv, y_cv)
            if score > maxScore:
                maxScore = score
                maxEsti = i
                maxFeat = j
            print "(estimators,max_features) =   "+ str((i,j)) + "   score:  "+ str(score)

    score = clf.score(X_test, y_test)
    print "(estimators,max_features) =   "+ str((maxEsti,maxFeat)) + "   score:  "+ str(score)

def doPCA(x,y):
    pca = PCA(n_components = 4)
    x = pca.fit_transform(x)
    print (pca.explained_variance_ratio_)
    return x,y

def display(x,y):
    h = .02 #step size in mesh

    
    datasets = [(x,y)]
    figure = plt.figure(figsize = (9,9))
    i = 1

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
     "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA()]
    for ds in datasets:
        X,y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1
    figure.subplots_adjust(left=.02, right=.98)
    plt.show()

if __name__ == '__main__':
    #(lm,invalid) = landmark()
    f1 = open("landmark_2500.txt","r")
    lm = pickle.load(f1)
    invalid = pickle.load(open("invalid_2500.txt","r"))

    x,y = getTraining(lm,invalid)
    x,y = doPCA(x,y)
    boost(x,y)
    #display(x,y)
    #parameterRBF(x,y)
