import numpy
from urllib.request import urlopen
from sklearn.metrics import mean_squared_error
import scipy.optimize
import random
import csv
from collections import defaultdict
from sklearn import linear_model
'''
a: 2.9
e: 0.16
i: 9
om: 168.5
w: 181
q: 2.4
ad: 3.38
per_y: 6.86
n_obs: 259
n: 0.24
ma: 180.66
'''
def readCSV(path):
    f = open(path, 'rt', encoding = 'UTF-8')
    c = csv.reader(f)
    header = next(c)
    for l in c:
        d = dict(zip(header,l))
        yield d

print("Reading data...")
# Linear regression
data = list(readCSV("Asteroid_Positive.csv"))
def feature(datum):
    if datum["a"].strip() == "":
        datum["a"] = 2.9
    if datum["e"].strip() == "":
        datum["e"] = 0.16
    if datum["i"].strip() == "":
        datum["i"] = 9
    if datum["om"].strip() == "":
        datum["om"] = 168.5
    if datum["w"].strip() == "":
        datum["w"] = 181
    feat = [1, float(datum["a"]), float(datum["e"]), float(datum["i"]), float(datum["om"]),float(datum["w"])]
    return feat
'''
X_train = []
y_moid = []
y_H = []
X_pred = []
i = 0
for d in data:
    if d["H"].strip() == "" or d["moid"].strip() == "":
        X_pred.append(feature(d))
        continue
    X_train.append(feature(d))
    y_moid.append(float(d["moid"]))
    y_H.append(float(d["H"]))
print("finished")
theta,residuals,rank,s = numpy.linalg.lstsq(X_train, y_H)
print(theta)
X_pred = numpy.matrix(X_pred)
y_pred_H =numpy.dot(theta,X_pred.T)
print(y_pred_H)
'''
# logistic regression
X_pred = []
X_train = []
y_train = []
y_pred = []
for d in data:
    if d["pha"].strip() == "":
        X_pred.append(feature(d))
        continue
    X_train.append(feature(d))
    y_train.append(d["pha"])
mod = linear_model.LogisticRegression(class_weight = "balanced")
mod.fit(X_train, y_train)
y_pred = mod.predict(X_pred)
print(y_pred)
'''
TP_ = numpy.logical_and(pred, y)
FP_ = numpy.logical_and(pred, numpy.logical_not(y))
TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(y))
FN_ = numpy.logical_and(numpy.logical_not(pred), y)
TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)
accuracy=(TP + TN) / (TP + FP + TN + FN)
print("accuracy for logistic: ",accuracy)
'''

