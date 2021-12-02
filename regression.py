import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import classification_report, plot_confusion_matrix, plot_roc_curve 

df = pd.read_csv('Asteroid_Positive.csv', low_memory=False)
df = df.drop(['name','diameter', 'albedo','rot_per','neo','condition_code','class'], axis=1)
df = df.dropna()
df['pha'] = df['pha'].map({'Y': 1, 'N': 0})
df = pd.get_dummies(df) 
X = df.drop('pha', axis=1)
y = df['pha']
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3,random_state=0)
'''
#Logistic Regression
clf1 = LogisticRegression(max_iter=10**4, C = 1, class_weight = 'balanced')
clf1.fit(X_train,y_train)
y_val_pred = clf1.predict(X_val)
print('accuracy of base model: ',accuracy_score(y_val,y_val_pred))
print('f1_score of base model: ',f1_score(y_val,y_val_pred,average=None))
print(classification_report(y_val,y_val_pred))
plot_confusion_matrix(clf1, X_val, y_val)
plot_roc_curve(clf1, X_val, y_val)
#XGB
clf1 = xgb.XGBClassifier()
clf1.fit(X_train,y_train)
y_val_pred = clf1.predict(X_val)
print('accuracy of xgboost: ',accuracy_score(y_val,y_val_pred))
print('f1_score of xgboost: ',f1_score(y_val,y_val_pred,average=None))
print(classification_report(y_val,y_val_pred))
plot_confusion_matrix(clf1, X_val, y_val)
feat_importances = pd.DataFrame(clf1.feature_importances_, index=X.columns)
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.title('Feature Importances')
sns.barplot(data = feat_importances.sort_values(0).T)
plot_roc_curve(clf1, X_val, y_val)
#decisiontree
clf1 = DecisionTreeClassifier()
clf1.fit(X_train,y_train)
y_val_pred = clf1.predict(X_val)
print('accuracy of decisiontree: ',accuracy_score(y_val,y_val_pred))
print('f1_score of decisiontree: ',f1_score(y_val,y_val_pred,average=None))
print(classification_report(y_val,y_val_pred))
plot_confusion_matrix(clf1, X_val, y_val)
feat_importances = pd.DataFrame(clf1.feature_importances_, index=X.columns)
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.title('Feature Importances')
sns.barplot(data = feat_importances.sort_values(0).T)
plot_roc_curve(clf1, X_val, y_val)
#random forest
clf1 = RandomForestClassifier()
clf1.fit(X_train,y_train)
y_val_pred = clf1.predict(X_val)
print('accuracy of randomforest: ',accuracy_score(y_val,y_val_pred))
print('f1_score of randomforest: ',f1_score(y_val,y_val_pred,average=None))
print(classification_report(y_val,y_val_pred))
plot_confusion_matrix(clf1, X_val, y_val)
feat_importances = pd.DataFrame(clf1.feature_importances_, index=X.columns)
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.title('Feature Importances')
sns.barplot(data = feat_importances.sort_values(0).T);
plot_roc_curve(clf1, X_val, y_val)
#AdaBoost
clf1 = AdaBoostClassifier()
clf1.fit(X_train,y_train)
y_val_pred = clf1.predict(X_val)
print('accuracy of AdaBoost: ',accuracy_score(y_val,y_val_pred))
print('f1_score of AdaBoost: ',f1_score(y_val,y_val_pred,average=None))
print(classification_report(y_val,y_val_pred))
plot_confusion_matrix(clf1, X_val, y_val)
feat_importances = pd.DataFrame(clf1.feature_importances_, index=X.columns)
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.title('Feature Importances')
sns.barplot(data = feat_importances.sort_values(0).T)
plot_roc_curve(clf1, X_val, y_val)
'''
#KNN
clf1 = KNeighborsClassifier()
clf1.fit(X_train,y_train)
y_val_pred = clf1.predict(X_val)
print('accuracy of KNeighborsClassifier: ',accuracy_score(y_val,y_val_pred))
print('f1_score of KNeighborsClassifier: ',f1_score(y_val,y_val_pred,average=None))
print(classification_report(y_val,y_val_pred))
plot_confusion_matrix(clf1, X_val, y_val)
feat_importances = pd.DataFrame(clf1.feature_importances_, index=X.columns)
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.title('Feature Importances')
sns.barplot(data = feat_importances.sort_values(0).T)
plot_roc_curve(clf1, X_val, y_val)
