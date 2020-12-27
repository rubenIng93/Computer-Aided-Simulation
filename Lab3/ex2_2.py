import csv
import hashlib
import time
import math
from pympler import asizeof
import numpy as np
import pandas as pd
import random
from scipy.stats import t

## a function which read the file and store it
# in a set of words it returns
# the set and the number of words
def read_dictionary(dict_file):
    words_set = set() # initialize the set
    counter = 0 # initialize the counter
    with open(dict_file, "r") as txt: # open the file
        words = csv.reader(txt)
        for line in words:
            words_set.add(line[0])
            counter += 1
    return words_set, counter

# input parameters
x = [i for i in range(19, 25)] # define all the given exponents
words_set, w = read_dictionary('Lab3/words_alpha.txt')# import the dictionary
debug = True
bs_fp_probabilities = [] # list that tracks the false positive prob for bitstring
bs_actual_size_list = [] # list that tracks the actual sizes for bitstring

print('\n***START ENGINE***\n')
print(f'The dictionary file contains {w} words')

for bits in x:

    bitstring = np.zeros(2 ** bits, dtype=bool) # initialize the bitstring with 'bits' bits
    

    for word in words_set:

        word_hash = hashlib.md5(word.encode("utf-8")) # md5 hash
        word_hash = int(word_hash.hexdigest(), 16) # cast integer format
        h = word_hash % 2 ** bits # map into a given range [0, 2^b - 1] for bitstring
        bitstring[h] = 1 # add the hash in the set
    
    # analytical approach
    bs_fp_probability = (np.sum(bitstring) / 2 ** bits) * 100 # compute the prob of false positive for bitstring array in %
    bs_fp_probabilities.append(bs_fp_probability) # append in the list
    bs_size = asizeof.asizeof(bitstring) # compute the size in bytes
    bs_actual_size_list.append(bs_size) # append the size in the list

    if debug:
        print(f'Bitstring length = {len(bitstring)}')
        print(f'P(FP) = {bs_fp_probability:.4f}% in a bitstring array whose shape is {bits}')
        print(f'Actual size for the bitstring with {bits} bits = {bs_size/1024:.2f}kB')


print('\n***PROCESS TERMINATED***\n')

print('***GENERATE A FILE***\n')

datafile_sim = open(f"Lab3/analytical.dat", "w") # open an empty file
print("#bits\tp(FP)\tsize(KB)", file=datafile_sim)
for i in range(len(x)):
    print(x[i], # nbits
        bs_fp_probabilities[i], # false positive prob bitstring
        bs_actual_size_list[i]/1024, # actual size bitstring in kB
        sep="\t", file=datafile_sim)
datafile_sim.close()

print('***FILE GENERATED***')



