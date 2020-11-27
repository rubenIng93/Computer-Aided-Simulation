import csv
import hashlib
import time

# read the txt file
w = 0 # initialize the counter
collision = False


with open("Lab3/words_alpha.txt", "r") as txt:
    words = csv.reader(txt)
    for line in words:
        w += 1

start = time.time()
with open("Lab3/words_alpha.txt", "r") as txt:
    words = csv.reader(txt)        
    for b in range(1, w):
            collision = False
            words_set = set() # initialize the set
            for word in words:
                word_hash = hashlib.md5(word[0].encode("utf-8")) # md5 hash
                word_hash = int(word_hash.hexdigest(), 16) # cast integer format
                h = word_hash % 2**b # map into a given range [0, 2^b - 1]
                if h not in words_set:
                    words_set.add(h) # add h in the set
                else:
                    collision = True
                    print("A collision has been experienced with {} bits for the fingerprinting".format(b))
                    break
                

end = time.time()
print("Execution time: {:.4f} s".format(end-start))