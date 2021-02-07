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
debug = False
# generation of the parameter x
b = [19, 20, 21, 22, 23, 24]
ci_level = 0.95
runs =100

def retrieve_ci(mean, stddev):
    t_treshold = t.ppf((ci_level + 1) / 2, df= runs-1)
    ci = t_treshold * stddev /np.sqrt(runs)
    return ci

def get_k_opt(n, m):
    k_opt = int((2**n)/m*math.log(2))
    if k_opt == 0:
        k_opt = 1
    return k_opt

def compute_all_hashes(md5, num_hashes, b):
    bits_to_update = []
    if (b+3*num_hashes>128):
        print('Error - at most 32 hashes')
        return -1
    for _ in range(num_hashes):
        value = md5 % (2**b)
        bits_to_update.append(value)
        md5 = md5 // (2**3)
    return bits_to_update


with open("Lab3/words_alpha.txt", "r") as txt:
    words = csv.reader(txt)
    flag = False
    for line in words:
        words_set.add(line[0])
        if flag == False and w > 19500:
            if asizeof.asizeof(words_set) > 1048576:
                one_MB_bound = w
                flag = True
        w += 1

print("We have to deal with {} words".format(w))

start = time.time()
false_positive_prob_means = []
false_positive_prob_cis = []
bs_fp_probabilities_analytical = []
bf_false_positive_prob_means = []
bf_false_positive_prob_cis = []
size_list = []
bloom_size_list = []
k_opt_list = []
theorical_fp_prob_list = []
theoretical_bf_size_list = []

random.seed(seed)

for n_bits in b:

    #compute the k_opt for each n. of bits
    k_opt = get_k_opt(n_bits, w)
    k_opt_list.append(int(k_opt)) #rounded at the first integer

    theorical_fp_prob = (1 - math.e ** (-k_opt*w/2**n_bits))**k_opt # for bloom filter
    theorical_fp_prob_list.append(theorical_fp_prob*100)
    
    bit_string_array = np.zeros(2**n_bits, dtype=bool) # initialize the bitstring array with all zero
    bloom_filter = np.zeros(2**n_bits, dtype=bool) # initialize the bloom filter with all zero

    for word in words_set:
        word_hash = hashlib.md5(word.encode("utf-8")) # md5 hash
        word_hash = int(word_hash.hexdigest(), 16) # cast integer format
        h = word_hash % 2**n_bits # map into a given range [0, 2^b - 1] for bitstring
        all_bits_to_update = compute_all_hashes(word_hash, k_opt, n_bits) # get all the bit to set to 1 in the bloom filter
        bit_string_array[h] = 1 # add h in the set
        bloom_filter[all_bits_to_update] = 1 # set the bits to 1 for BF

    # test prob of false positive
    bs_fp_probability = (np.sum(bit_string_array) / 2 ** n_bits) * 100 # analytical fp prob bitstring
    analytical_prob_bf = (np.sum(bloom_filter)/(2**n_bits))**k_opt # analytical fp prob bloom filter
    ba_size = asizeof.asizeof(bit_string_array) # get the size in bytes
    
    bs_fp_probabilities_analytical.append(bs_fp_probability) # append in the list
    size_list.append(ba_size/1024) # append it in order to save the output in kb
    
    run_means = np.zeros(runs) # list with means of the runs
    bf_run_means = np.zeros(runs) # list with means of the runs for BF

    for run in range(runs):
        fp_counter = 0 # false positive counter for bitstring
        fp_bf_counter = 0 # false positive counter for bloof filter
        for _ in range(fake_attempt):
            wrong_h_list = []
            for _ in range(k_opt): # generate k_opt fake hash
                wrong_h = random.randint(0, 2**n_bits-1) # generate a fake hash
                wrong_h_list.append(wrong_h)         

            if bit_string_array[wrong_h_list[0]] == 1: # test bitstring
                fp_counter += 1 # conflitc happens
            if bloom_filter[wrong_h_list].all() == 1: # all the bits are 1 in bloom filter
                fp_bf_counter += 1 # conflitc happens
            
        run_means[run] = fp_counter/fake_attempt # save the run's mean
        bf_run_means[run] = fp_bf_counter/fake_attempt

    # bitstring stast
    mean = np.mean(run_means) # get the mean of all the runs
    std = np.std(run_means, ddof=1) # get the standar deviation of all the runs
    ci = retrieve_ci(mean, std) # retrieve the CI
    false_positive_prob_means.append(mean*100)
    false_positive_prob_cis.append(ci*100)

    #bloom filter stats
    mean_bf = np.mean(bf_run_means) # get the mean of all the runs
    std_bf = np.std(bf_run_means, ddof=1) # get the standar deviation of all the runs
    ci_bf = retrieve_ci(mean_bf, std_bf) # retrieve the CI
    bf_false_positive_prob_means.append(mean_bf*100)
    bf_false_positive_prob_cis.append(ci_bf*100)
    
    bf_size = asizeof.asizeof(bloom_filter) # get the BF size in bytes
    bloom_size_list.append(bf_size/1024) # append the size of BF in kb
    theoretical_bf_size = w * 1.44 * math.log2(1/theorical_fp_prob) # theoretical size for bloom filter in bits
    theoretical_bf_size_list.append(theoretical_bf_size/(1024*8)) # append in kb


    if debug:
        print('>>>BITSTRING')
        print(f'Bitstring fp for {n_bits}bits = {mean*100}% (mean value)')
        print(f'Bitstring analytical fp prob. for {n_bits} = {bs_fp_probability}%')
        print(f'Bitstring actual size = {ba_size/1024} kb')
        
        print('>>>BLOOM FILTER')
        print(f'Bloom filter analytical fp for {n_bits}bits = {analytical_prob_bf*100}% (mean value)')
        print(f'Bloom filter fp for {n_bits}bits = {mean_bf*100}% (mean value)')
        print(f'Bloom filter theory fp for {n_bits}bits = {theorical_fp_prob*100}%')
        print(f'Bloom filter theoretical size = {theoretical_bf_size/(1024*8)} kb')
        print(f'Bloom filter actual size = {bf_size/1024} kb')
        print('\n')


    #print(f"P(FP) = {fp_counter/fp_attempt*100:.2f}% for 2^{n_bits} bits allocated")
    #print(f"Bit String Array size with 2^{n_bits} bits allocated = {ba_size/1024:.2f} KB")
    #print(f"Bloom filter size with 2^{n_bits} bits allocated = {bf_size/1024:.2f} KB")

# write a file with the measures obtained
datafile_sim = open(f"Lab3/lab3{runs}runs.dat", "w") # open an empty file
print("#bits\tanalyticalFPBS\tciLow\tP(FP)\tciHigh\tsize(KB)\tkopt\tBFciLow\tBFp(FP)\tBFciHigh\tTHfpProb\tBFsize(KB)", file=datafile_sim)
for i in range(len(b)):
    print(b[i], # nbits
        bs_fp_probabilities_analytical[i], # fp prob analytical approach bitstring
        false_positive_prob_means[i] - false_positive_prob_cis[i], # ccLow bitstring
        false_positive_prob_means[i], # mean bitstring
        false_positive_prob_means[i] + false_positive_prob_cis[i], # ciHigh bitstring
        size_list[i], # size bitstring
        k_opt_list[i],  # optiman n. hash bloom filter
        bf_false_positive_prob_means[i] - bf_false_positive_prob_cis[i], # ciLow bloom filter 
        bf_false_positive_prob_means[i], # mean bloom filter
        bf_false_positive_prob_means[i] + bf_false_positive_prob_cis[i], # ciHigh bloom filter
        theorical_fp_prob_list[i], # theoretical fp prob for bloom filter
        bloom_size_list[i], # bloom filter size
        theoretical_bf_size_list[i], # theoretical bf size
        sep="\t", file=datafile_sim)
datafile_sim.close()

# display result in a pandas dataframe
print("\nTo resume:\n")
pd.set_option('display.float_format', lambda x: '%.8f' % x)
df = pd.DataFrame(data={
                        'Bits per fingerprint':b,
                        'Optimal # hash functions:':k_opt_list,
                        'Prob. false positive [%]':theorical_fp_prob_list,
                        'Min theoretical memory [KB]':theoretical_bf_size_list,
                        'Actual memory [KB]':bloom_size_list}
                )

print(df.set_index('Bits per fingerprint'))

# question 9
#
# 1 MB = 1024 KB = 1048576 Bytes = 8388608 bits
# 
# reversing the formula for bitstring we have
# b = log2(n) where n is the total storage for bitstring array
# 
# same for bloom filter#

one_MB_bits = math.log2(8388608)

print(f"\n\nHaving 1MB available for the dictionary set membership problem\n\
    corresponds to the case with {one_MB_bits} bits for the fingerprinting\n")

# Computation for bits allocated for fingerprint table
# 8388608 bits = 370103 * b
b_finger = 8388608 / w
b_finger = math.floor(b_finger) # round to the floor integer
print('\nBits allocated for each fingerprint stored in a table = ', b_finger)
n = 2 ** b_finger # get n 
finger_fp_prob = (1-(1-(1/n))**w) * 100 # compute the false positive probability

print('\n')
df = pd.DataFrame(data={
                        'Storage':['Words set', 'Fingerprint set', 'Bitstring array', 'Bloom filter'],
                        'Prob. false positive':[f'It stores {one_MB_bound} words', str(round(finger_fp_prob, 4))+'%', '4.3161%', '0.0019%'],
                        }
                )

print(df.set_index('Storage'))





