import numpy as np
import random
import math
from scipy.stats import t


# VARIABLES
n_people = 365 # n in the theory 
# number of people amonge the selection can be made
seed = 22
ci_level = 0.95 # level of the confidence interval
runs = 1000 # number of runs to build the ci
stop_value = 2.5*math.sqrt(n_people)

# data structures
conflict_per_run = np.full(0, runs) # wheter or not a conflict has been experienced

def evaluate_ci(mean, std):
    t_treshold = t.ppf((ci_level + 1) / 2, df= runs-1)
    ci = t_treshold * stddev /np.sqrt(runs)
    #rel_err = ci / mean
    rel_err = 0 # must be restored
    return ci, rel_err


# main engine
for run in range(runs):
    chosen_day = np.full(0, n_people) # array that track the conflicts
    for guy in range(stop_value): # more is always 1 or close to it
        date = random.randint(0, n_people-1)
        if chosen_day[date] == 1: # there is already a guy here
            conflict_per_run[run] = 1
            break
        else:
            chosen_day[date] = 1 # occupy the date

prob_conflict = np.sum(conflict_per_run)/runs # also the mean
std = math.sqrt(runs * prob_conflict * (1 - prob_conflict)) # from binomial distr
ci, rel_err = evaluate_ci(prob_conflict, std) # get the ci
ciLow = prob_conflict - ci
ciHigh = prob_conflict + ci

# save in a file
datafile = open(f"Lab2/data/BP{n_people}.dat", 'r')
print("m", file=datafile)
