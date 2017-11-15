#!/usr/bin/python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi". 

features_list = ['poi', "exercised_stock_options", "total_stock_value", "deferred_income"] #, "salary" , "exercised_stock_options", "to_poi_ratio", "bonus"
                 
### Load the dictionary containing the dataset  "bonus", "to_poi_ratio"
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# load the data
data = pd.DataFrame.from_dict(data_dict, orient="index")
df = data

# convert all the features except from email_address to floats
for col in df:
    if col != "email_address":
        df[col] = df[col].apply(float, 1) 

# basic info about dataset
no_data_points = len(df)
no_features = len(df.columns)

#sns.set(style="whitegrid")
#g = sns.factorplot("poi", data=df, kind="count")
#
#sns.lmplot("bonus", "total_stock_value", data=df, fit_reg=False, hue="poi")
#plt.xlabel("bonus")
#plt.ylabel("total_stock_value")
#plt.show()
#
#sns.lmplot("total_payments", "exercised_stock_options", data=df, fit_reg=False, hue="poi")
#plt.xlabel("total_payments")
#plt.ylabel("exercised_stock_options")
#plt.show()

#count number of nans within each feature
nans_f = pd.Series(df.isnull().sum(axis=0), name="nans")
nans_f = nans_f.sort_values(ascending=False)
nans_o = pd.Series(df.isnull().sum(axis=1), name="nans")
nans_o = nans_o.sort_values(ascending=False)

# print the features after droping the outliers
outliers = df.nlargest(3, "total_payments")
df = data.drop(["email_address", "other", "loan_advances"], 1)
df = df.drop(["TOTAL"])

#sns.lmplot("bonus", "total_stock_value", data=df, fit_reg=False, hue="poi")
#plt.xlabel("bonus")
#plt.ylabel("total_stock_value")
#plt.show()
#
#sns.lmplot("total_payments", "exercised_stock_options", data=df, fit_reg=False, hue="poi")
#plt.xlabel("total_payments")
#plt.ylabel("exercised_stock_options")
#plt.show()

#g = sns.PairGrid(df_scaled, hue="poi", palette="Set1", dropna=True).map(plt.scatter)
#g.savefig("pairplot.png")
#
#f, ax = plt.subplots(figsize=(11, 9))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
#corr = df.corr()
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0,
#            square=True, linewidths=.5, cbar_kws={"shrink": .5})


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# creating a new features
nans = pd.Series(df.isnull().sum(axis=1), name="nans")
df = df.join(nans)

to_poi_ratio = pd.Series(df["from_this_person_to_poi"]/df["from_messages"], \
                         name="to_poi_ratio")
df = df.join(to_poi_ratio)
from_poi_ratio = pd.Series(df["from_poi_to_this_person"]/df["to_messages"], \
                           name="from_poi_ratio")
df = df.join(from_poi_ratio)

# feature scaling
df["director_fees"] = df["director_fees"].fillna(0)
from sklearn import preprocessing
imp = preprocessing.Imputer(missing_values=np.nan, strategy='median', axis=0)
scaler = preprocessing.MinMaxScaler()
df_imputed = pd.DataFrame(imp.fit_transform(df), \
                          index=df.index, columns=df.columns)
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), \
                         index=df.index, columns=df.columns)
poi = df["poi"] 
df_scaled["poi"] = poi

# feature selection
from sklearn.feature_selection import SelectKBest, f_classif, chi2
selector1 = SelectKBest(f_classif)
selector2 = SelectKBest(chi2)
top_features1 = selector1.fit_transform(df_scaled, df_scaled["poi"])
top_features2 = selector2.fit_transform(df_scaled, df_scaled["poi"])
mask1 = selector1.get_support()
mask2 = selector2.get_support()
columns1 = []
columns2 = []

# creates lists of scores for all the features
for col in range(len(df.columns)):
    if mask1[col] == True:
        columns1.append(df.columns[col])
    if mask2[col] == True:
        columns2.append(df.columns[col])
        
sc1 = pd.DataFrame(selector1.scores_, columns=["anova"], index=df.columns)
sc1 = sc1.sort_values(by=["anova"], ascending=False)
sc2 = pd.DataFrame(selector2.scores_, columns=["chi2"], index=df.columns)
sc2 = sc2.sort_values(by=["chi2"], ascending=False)

# transforming back to original dictionary format
df = df.fillna("NaN")
#data_dict = df.to_dict("index")
data_dict = df_scaled.to_dict("index")
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# creation of classificators
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()

from sklearn import svm
svma = svm.SVC()

from sklearn import tree
dt = tree.DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# parameters tuning using the simple train_test_split
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, recall_score
parameters_ada = {"n_estimators": range(10, 160, 10), "learning_rate" : [1.0, 1.5, 2.0]}
ada_opt = GridSearchCV(ada, parameters_ada)
ada_opt.fit(features_train, labels_train)
print "adaboost parameters: ",  ada_opt.best_params_
print "adaboost accuracy: ",  ada_opt.best_estimator_.score(features_test, labels_test)
#print "adaboost best estimator: ",  ada_opt.best_estimator_
print "adaboost precision score: ",  precision_score(labels_test, ada_opt.predict(features_test))
print "adaboost recall score: ",  recall_score(labels_test, ada_opt.predict(features_test))

parameters_svm = {"kernel" : ["linear", "rbf"],
                   "C" : [1, 10, 30, 50, 70, 80, 90, 100, 120, 150, 1000],
                   "gamma" : [0, 0.001, 0.00001]}
svma_opt = GridSearchCV(svma, parameters_svm)
svma_opt.fit(features_train, labels_train)
print "SVM parameters: ", svma_opt.best_params_
print "SVM accuracy: ",  svma_opt.best_estimator_.score(features_test, labels_test)
#print "SVM best estimator: ",  svma_opt.best_estimator_
print "SVM precision score: ",  precision_score(labels_test, svma_opt.predict(features_test))
print "SVM recall score: ",  recall_score(labels_test, svma_opt.predict(features_test))

parameters_tree = {"min_samples_split" : range(2, 101)}
dt_opt = GridSearchCV(dt, parameters_tree)
dt_opt.fit(features_train, labels_train)
print "decision tree parameters: ", dt_opt.best_params_
print "decision tree accuracy: ",  dt_opt.best_estimator_.score(features_test, labels_test)
#print "decision tree best estimator: ",  dt_opt.best_estimator_
print "decision tree precision score: ",  precision_score(labels_test, dt_opt.predict(features_test))
print "decision tree recall score: ",  recall_score(labels_test, dt_opt.predict(features_test))

nb.fit(features_train, labels_train)
print "naive bayes accuracy: ", nb.score(features_test, labels_test)
print "naive bayes precision score: ",  precision_score(labels_test, nb.predict(features_test))
print "naive bayes recall score: ",  recall_score(labels_test, nb.predict(features_test))


# parameters tuning using StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedShuffleSplit
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(labels, 1000, random_state = 1)
for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
    
    
ada_opt = GridSearchCV(ada, parameters_ada)
ada_opt.fit(features_train, labels_train)
print "adaboost parameters: ",  ada_opt.best_params_
print "adaboost accuracy: ",  ada_opt.best_estimator_.score(features_test, labels_test)
#print "adaboost best estimator: ",  ada_opt.best_estimator_
print "adaboost precision score: ",  precision_score(labels_test, ada_opt.predict(features_test))
print "adaboost recall score: ",  recall_score(labels_test, ada_opt.predict(features_test))

svma_opt = GridSearchCV(svma, parameters_svm)
svma_opt.fit(features_train, labels_train)
print "SVM parameters: ", svma_opt.best_params_
print "SVM accuracy: ",  svma_opt.best_estimator_.score(features_test, labels_test)
#print "SVM best estimator: ",  svma_opt.best_estimator_
print "SVM precision score: ",  precision_score(labels_test, svma_opt.predict(features_test))
print "SVM recall score: ",  recall_score(labels_test, svma_opt.predict(features_test))

dt_opt = GridSearchCV(dt, parameters_tree)
dt_opt.fit(features_train, labels_train)
print "decision tree parameters: ", dt_opt.best_params_
print "decision tree accuracy: ",  dt_opt.best_estimator_.score(features_test, labels_test)
#print "decision tree best estimator: ",  dt_opt.best_estimator_
print "decision tree precision score: ",  precision_score(labels_test, dt_opt.predict(features_test))
print "decision tree recall score: ",  recall_score(labels_test, dt_opt.predict(features_test))

nb.fit(features_train, labels_train)
print "naive bayes accuracy: ", nb.score(features_test, labels_test) 
print "naive bayes precision score: ",  precision_score(labels_test, nb.predict(features_test))
print "naive bayes recall score: ",  recall_score(labels_test, nb.predict(features_test))

# choosing of classifier used in final evaluation
clf = nb

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
