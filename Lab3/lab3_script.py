import csv
import hashlib
import time
import math
from pympler import asizeof
import pandas as pd

##
# FINGERPRINTING HERE#


# read the txt file
w = 0 # initialize the counter
words_set = set()
epsilon = 0.5

with open("Lab3/words_alpha.txt", "r") as txt:
    words = csv.reader(txt)
    for line in words:
        words_set.add(line[0])
        w += 1

print("We have to deal with {} words".format(w))

start = time.time()

fingerprint_set = set() # initialize the set       
#size_empty_set = asizeof.asizeof(fingerprint_set)
for b in range(20, w): # 20 is a good starting number
    for word in words_set:
        word_hash = hashlib.md5(word.encode("utf-8")) # md5 hash
        word_hash = int(word_hash.hexdigest(), 16) # cast integer format        
        h = word_hash % 2**b # map into a given range [0, 2^b - 1]
        if h in fingerprint_set:
            print("A collision has been experienced with {} bits for the fingerprinting".format(b))
            #print("The set contains: ",len(fingerprint_set))
            fingerprint_set.clear()
            break 
        else:
            fingerprint_set.add(h) # add h in the set

    if len(fingerprint_set) == w:
        print("Mininum #bits for no conflict = {}bits".format(b))
        b_exp = b
        break            
        
# the b^theo is computed as log2(m/epsilon)
# where m is the number of elements i want to store
# and epsilon a given probability
b_theo = math.log((w / 0.5), 2) # theoretical bits requirements
print("\nThe number of bits needed for get a probability\n of fingerprint collision < 0.5 is {} bits".format(int(b_theo)))


# THEORETICAL STORAGE
# for the fingerprint set we have to store b_exp bits for each word
# a good avg size for a string could be 50 bytes
# avg chars of English words is 4.7; 1 char = 1 byte = 8 bit https://wolfgarbe.medium.com/the-average-word-length-in-english-language-is-4-7-35750344870f
fing_set_theory_storage = w * b_exp / 8 # this bytes
words_set_theory_storage = w * 4.7  # this is in bytes
print("\nTheoretical storage for words set = {:.2f} KB".format(words_set_theory_storage/1024))
print("Theoretical storage for {}-fingerprint set = {:.2f} KB".format(b_exp, fing_set_theory_storage/1024))

# ACTUAL STORAGE
# first put the word in a words set
string_set_storage = asizeof.asizeof(words_set)
hash_set_storage = asizeof.asizeof(fingerprint_set)
print("Storing the strings in a set takes: {:.2f} KB".format(string_set_storage/1024))
print("Storing the fingerprints in a set takes: {:.2f} KB".format(hash_set_storage/1024))
print("With a space saving of {:.2f}x".format(string_set_storage/hash_set_storage))

# P(FP) = 1-(1-(1/n))^m
n = 2**b_exp
p_false_positive = 1-(1-(1/n))**w
print("\nWith the Fingerprint whose range is [0, 2^{}-1] P(FP) = {:.6f}%".format(b_exp, p_false_positive*100))

print("\nThe set contains: ",len(fingerprint_set))
end = time.time()
print("Execution time: {:.2f} s".format(end-start))

print("\nTo resume:\n")
df = pd.DataFrame(data={'Storage':['Word set', 'Fingerprint set'],
                        'Bits per fingerprint':['N.A.', b_exp],
                        'Prob. false positive [%]':['N.A.', p_false_positive*100],
                        'Min theoretical memory [KB]':[words_set_theory_storage/1024, fing_set_theory_storage/(1024)],
                        'Actual memory [KB]':[string_set_storage/1024, hash_set_storage/1024]}, 
                )

print(df.set_index('Storage'))



