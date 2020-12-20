import csv
import hashlib
import time
import math
from pympler import asizeof
import numpy as np
import pandas as pd
import random
from scipy.stats import t

# read the txt file
w = 0 # initialize the counter
words_set = set()
epsilon = 0.5
fake_attempt = 10
seed = 22
runs = 10
# generation of the parameter x
b = [19, 20, 21, 22, 23, 24]
ci_level = 0.95
runs =100

def retrieve_ci(mean, stddev):
    t_treshold = t.ppf((ci_level + 1) / 2, df= runs-1)
    ci = t_treshold * stddev /np.sqrt(runs)
    return ci


with open("Lab3/words_alpha.txt", "r") as txt:
    words = csv.reader(txt)
    for line in words:
        words_set.add(line[0])
        w += 1

print("We have to deal with {} words".format(w))

start = time.time()
false_positive_prob_means = []
false_positive_prob_cis = []
size_list = []

random.seed(seed)

for n_bits in b:
    
    bit_string_array = np.zeros(2**n_bits) # initialize the bitstring array with all zero
    for word in words_set:
        word_hash = hashlib.md5(word.encode("utf-8")) # md5 hash
        word_hash = int(word_hash.hexdigest(), 16) # cast integer format
        h = word_hash % 2**n_bits # map into a given range [0, 2^b - 1]
        bit_string_array[h] = 1 # add h in the set
    # test prob of false positive
    ba_size = asizeof.asizeof(bit_string_array) # get the size
    size_list.append(ba_size) # append it in order to save the output
    run_means = np.zeros(runs) # list with means of the runs
    for run in range(runs):
        fp_counter = 0
        for _ in range(fake_attempt):
            wrong_h = random.randint(0, 2**n_bits-1) # generate a fake hash
            if bit_string_array[wrong_h] == 1:
                fp_counter += 1 # conflitc happens
        run_means[run] = fp_counter/fake_attempt # save the run's mean
    mean = np.mean(run_means) # get the mean of all the runs
    std = np.std(run_means, ddof=1) # get the standar deviation of all the runs
    ci = retrieve_ci(mean, std) # retrieve the CI
    false_positive_prob_means.append(mean*100)
    false_positive_prob_cis.append(ci*100)


    #print(f"P(FP) = {fp_counter/fp_attempt*100:.2f}% for 2^{n_bits} bits allocated")
    print(f"Bit String Array size with 2^{n_bits} bits allocated = {ba_size/1024:.2f} KB")

datafile_sim = open(f"Lab3/BitstringHash{runs}runs.dat", "w") # open an empty file
print("#bits\tciLow\tP(FP)\tciHigh", file=datafile_sim)
for i in range(len(b)):
    print(b[i], false_positive_prob_means[i] - false_positive_prob_cis[i],\
        false_positive_prob_means[i], false_positive_prob_means[i] + false_positive_prob_cis[i],\
            sep="\t", file=datafile_sim)
datafile_sim.close()



