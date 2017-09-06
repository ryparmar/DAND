# -*- coding: utf-8 -*-
"""
Udacity Nanodegree Program, Data Analyst

Project_3: 
            Data Analysis

Dataset:   
            Titanic
            
(Note:   Titanic dataset was adjusted as the original dataset on Kaggle consists of 2224 passengers 
        and the used dataset consists only of 891 passengers.)            

General question:
            What are the significant factors/characteristics of survivors?
            
Sub-questions:
            Did have childrens and womans better chances to survive?           
            Did people with better social status (higher class) had better probability of survival?
           
List of used websites, books, etc.
Pandas Documentation, NumPy Documentation, Python Documentation, Seaborn Documentation, Matplotlib Documentation
1) https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-the-column-in-panda-data-frame
            
            
"""




# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reading data file from csv
data_df = pd.read_csv("titanic-data.csv")

# first touch with the data
print data_df.head()


# -----------------------------------------------------------------------------
# DATA EXPLORATION
# -----------------------------------------------------------------------------


#sns.countplot(x="Survived", hue="Sex", data=data_df, orient="v")
#sns.set(style="darkgrid", color_codes=True)
#sns.lmplot(x="Age", y="Survived", data=data_df, logistic=True, y_jitter=.03)


#sns.countplot(x="Pclass", hue="Survived", data=data_df, orient="v")
#sns.lmplot(x="Age", y="Survived", hue="Sex", data=data_df, sharey=False)

#plt.plot(x=data_df["Age"], y=data_df["Survived"])

# correlation matrix preparations
# http://seaborn.pydata.org/examples/many_pairwise_correlations.html
#sns.set(style="white")
#corr = data_df.corr()
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#f, ax = plt.subplots(figsize=(11,9))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
#sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.4, center=0, square=True, linewidth=1)

#v = sns.factorplot(x="Pclass", y="Survived", data=data_df, kind="bar")


# sum up the NaN values
# 1)
count_nan = data_df.isnull().sum()
print count_nan

embarked = data_df.groupby(["Embarked", "Survived", "Sex"], as_index=False).count()
print embarked[["Embarked", "Survived","PassengerId", "Sex"]]

print data_df.describe()


# correlation matrix preparations
# http://seaborn.pydata.org/examples/many_pairwise_correlations.html
#sns.set(style="white")
corr = data_df.corr()
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#f, ax = plt.subplots(figsize=(11,9))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
#sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.4, center=0, square=True, linewidth=1)


"""
As I dont see any added value in below named features right now, I decided to remove them.
    Name column - just duplicate the PassengerID column,
    Ticket column - no simple idea how to use it
    Cabin - 687 out of 893 values are NaN
    Embarked - I dont believe that port of embarkation has any significant affect, whether passenger survive or not
"""
def drop_features(df):
    return df.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)
data_df = drop_features(data_df)


"""
Next, I have to solve the problem with NaN values. As I found earlier there are NaN values only in Age column now. 
As I do not want to loose the data in my sample I perform a transformation of this feature into more intuitive groups
by filling of the all NaN values by negative integer and then group the values according to age. 
"""

def transform_age(df):
    df["Age"] = df["Age"].fillna(-1)
    bins = [-2, 0, 6, 12, 18, 30, 60, 120]
    groups = ["Unknown", "Baby", "Child", "Young", "Younger Adult", "Older Adult", "Senior"]
    cut = pd.cut(df["Age"], bins, labels=groups)
    df["Age"] = cut
    return df

transform_age(data_df)


# https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/#loc-selection
# create new category in Sex column = Child
data_df.loc[data_df["Age"] == ("Baby" or "Child"), "Sex"] = "child"

# return proportion of variable on feature
# https://stackoverflow.com/questions/18062135/combining-two-series-into-a-dataframe-in-pandas
def get_proportions(df, feature):
    denominator =  df[[feature, "Survived"]].groupby([feature]).count()
    denominator2 = df["Survived"].sum()
    numerator = df[[feature, "Survived"]].groupby([feature]).sum()
    
    prop = np.round(numerator["Survived"].divide(denominator["Survived"])*100,2)
    prop.name = "Survived [%]"
    prop2 = np.round(numerator["Survived"].divide(denominator2)*100,2)
    prop2.name = "Survived / Total Survived [%]"
    proportions =  pd.concat([prop, prop2], axis=1)
    return proportions

proportion_age = get_proportions(data_df, "Age")


