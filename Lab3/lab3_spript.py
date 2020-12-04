import csv
import hashlib
import time
import math
from pympler import asizeof

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
for b in range(1, w):
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
b_theo = math.log((w / epsilon), 2) # theoretical bits requirements
print("\nThe number of bits needed for get a probability\n of fingerprint collision < 0.5 is {} bits".format(int(b_theo)))

# ACTUAL STORAGE
# first put the word in a words set
string_set_storage = asizeof.asizeof(words_set)
hash_set_storage = asizeof.asizeof(fingerprint_set)
print("\nStoring the strings in a set requires: {:.2f} KB".format(string_set_storage/1024))
print("Storing the fingerprints in a set requires: {:.2f} KB".format(hash_set_storage/1024))
print("With a space saving of {:.2f}x".format(string_set_storage/hash_set_storage))




print("\nThe set contains: ",len(fingerprint_set))
end = time.time()
print("Execution time: {:.2f} s".format(end-start))