# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 20:07:30 2017

@author: mrecl
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

"""
for 50 trials
The sample mean is 20.07. 
The sample median is 21. 
The sample standard deviation is 5.09. 
The sample variance is 25.86. The interquartile range is 7.
"""

#    deck = list(["Ac", "2c", "3c", "4c", "5c", "6c", "7c", "8c", "9c", "10c", "Jc", "Qc", "Kc",
#               "Ad", "2d", "3d", "4d", "5d", "6d", "7d", "8d", "9d", "10d", "Jd", "Qd", "Kd",
#               "Ah", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "Jh", "Qh", "Kh",
#               "As", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "10s", "Js", "Qs", "Ks"])
#    my_seed = np.chararray((trials, 3), itemsize=3)

# deck initialization
deck = list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10,])

# a histogram representing the absolute frequencies of the card values from a single draw
plt.hist(deck, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], align="left")
plt.title("Histogram of card deck")
plt.show()

# a histogram representing the relative frequencies of the card values from a single draw
plt.hist(deck, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], normed=True, align="left")
plt.title("Histogram of card deck")
plt.show()


# function generating card/numbers representating cards
def my_seed(trials):
    seq = range(trials)
    my_seed = np.empty((trials, 3), dtype=np.int)
    for row in seq:
        my_seed[row] = np.random.choice(deck, size = (3), replace=False)
    return my_seed

#def cardsToNumbers(draw):
    
seed = my_seed(50)
#print(seed)


# a histogram representing the absolute frequencies of the card values from a n draws
plt.hist(seed, bins=[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], stacked=1, align="left")
plt.title("Histogram of generated draws")
plt.show()

# a histogram representing the relative frequencies of the card values from a n draws
plt.hist(seed, bins=[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], normed=True, stacked=1, align="left")
plt.title("Histogram of generated draws")
plt.show()

# a histogram representing the relative frequencies of the summed card values from a n draws
seed_sum = np.sum(seed, axis=1)
plt.hist(seed_sum, bins=[ 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33], normed=True, stacked=1, align="left")
plt.title("Histogram of sums from generated draws")
plt.show()


# deck statistics / one draw statistics
deck_mean = np.round(np.mean(deck),2)
print("deck_mean =", deck_mean)
deck_median = np.round(np.median(deck),2)
print("deck_median =", deck_median)
deck_std = np.round(np.std(deck),2)
print("deck_standard_deviation =", deck_std)


# sample statistics calculations
seed_mean = np.mean(np.sum(seed, axis=1))
print("sample_mean= ", seed_mean)
seed_median = np.median(seed_sum)
print("sample_median= ", seed_median)
seed_std = np.round(np.std(seed_sum),2)
print("sample_standard_deviation= ", seed_std)
seed_var = np.var(seed_sum)
print("sample_variance= ", seed_var)
seed_iqr = np.percentile(seed_sum, q=75) - np.percentile(seed_sum, q=25)
print("sample_interquantile_range= ", seed_iqr)
print("in other words: 50% of values lies within this range: ", np.percentile(seed_sum, q=25), " - ", np.percentile(seed_sum, q=75))


# making some estimates
seed_est1 = np.percentile(seed_sum, q=95) - np.percentile(seed_sum, q=5)
print("based on this sample, I can say that with 90% probability can be approximately expected that future values will lie within this range: ", np.percentile(np.sum(seed, axis=1), q=5), " - ", np.percentile(np.sum(seed, axis=1), q=95))

#Q: What is the approximate probability that you will get a draw value of at least 20? 
#A_theroretically: 
#   Count all the combinations without replacement within deck (set of cards) which sum is at least 20 
#   and divide it by the amount of all the possible combinations within the deck
#A_practically:
#   Make an estimes based on sample - count the amount of all draws which sum is at least 20
#   and divide it by number of trials (total number of draws, 1 draw = 3 cards)

#rows = range(len(seed_sum))
#seed_est2 = np.zeros(len(seed_sum))
#for i in rows:
#    if (seed_sum[i] >= 20):
#        seed_est2[i] = 1
#    else:
#        seed_est2[i] = 0
#seed_est2 = np.sum(seed_est2)/len(seed_sum)
       
        
z_sc = (20 - seed_mean)/seed_std
seed_est2 = 1-st.norm.cdf(z_sc)
print("based on this sample the approximate probability that draw value will be at least 20 is: ",
      seed_est2)