# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("C:\Users\mrecl\Desktop\Inferential_Statistics\project\stroopdata.csv")

data["Difference"] = data["Congruent"] - data["Incongruent"]

avg_cong = round(np.average(data["Congruent"]), 2)
avg_incong = round(np.average(data["Incongruent"]), 2)
avg_diff = round(np.average(data["Difference"]), 2)

median_cong = round(np.median(data["Congruent"]), 2)
median_incong = round(np.median(data["Incongruent"]), 2)
median_diff = round(np.median(data["Difference"]), 2)

var_cong = round(np.var(data["Congruent"]), 2)
var_incong = round(np.var(data["Incongruent"]), 2)
var_diff = round(np.var(data["Difference"]), 2)

sd_cong = round(np.sqrt(var_cong), 2)
sd_cong = round(np.sqrt(var_incong), 2)
sd_diff = round(np.sqrt(var_diff), 2)

n = len(data)
dof = len(data) - 1

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 2, figsize=(10, 7), sharex = True)
sns.despine(left = True)

sns.distplot(data["Congruent"], ax=axes[0], bins=5)
sns.distplot(data["Incongruent"], color="r", ax=axes[1], bins=5)


#sns.set(style="white", palette="muted", color_codes=True)
#sns.distplot(data["Congruent"], color="b", bins=5, label="congruent", axlabel="")
#sns.distplot(data["Incongruent"], color="r", bins=5, label="incongruent", axlabel="")
#plt.legend()


t_stat = (avg_cong - avg_incong)/()
