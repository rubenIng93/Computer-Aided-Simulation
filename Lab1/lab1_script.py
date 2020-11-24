import numpy as np
import random
import time
from scipy.stats import t

# initial settings
seed = 2502
confidence_level = 0.95
runs = 40
debug = True

class BinNBalls_simulator:
    def __init__(self, runs, seed, confidence_level, debug):
        self.input_list = []
        self.runs = runs
        self.confidence_level = confidence_level
        self.debug = debug
        random.seed(a=seed)

    def create_inputs(self):
        for i in(2,3,4,5):
            a = [x*10**i for x in(1,2,4,8)]
            self.input_list.extend(a)
        self.input_list.extend([1000000])

    def get_inputs(self):
        return self.input_list

    def evaluate_conficence_interval(self, x):
        t_treshold = t.ppf((self.confidence_level + 1) / 2, df= self.runs-1)
        ave = x.mean()
        stddev = x.std(ddof=1)
        ci = t_treshold * stddev /np.sqrt(self.runs)
        rel_err = ci/ ave
        if self.debug:
            print("Min: {:.2f}, Ave: {:.2f}, Max: {:.2f}".format(x.min(), ave, x.max()))
        return ave, ci, rel_err

    def run(self, n):
        maxvec = np.full(self.runs, 0)
        for run in range(self.runs):
            bins = np.full(n, 0)
            for _ in range(n):
                bins[random.randint(0, n-1)] += 1
            maxvec[run] = bins.max()
        ave, ci, rel_err = self.evaluate_conficence_interval(maxvec)
        lower_bound = np.log(n) / np.log(np.log(n))
        return n, lower_bound, 3 * lower_bound, ave - ci, ave, ave + ci, rel_err

bin = BinNBalls_simulator(runs, seed, confidence_level, debug)
bin.create_inputs()


print("***INITIAL SETTINGS***")
print("Bins/Balls number for the simulation:")
print(bin.get_inputs())
print("Initial seed: ", seed)
print("Confidence level: ", confidence_level)
print("Number of runs: ", runs)
print("***END INITIAL SETTINGS***")

start = time.time()
print("<<<START SIMULATION>>>")

datafile = open("Lab1/binsballs"+str(runs)+"runs.dat", "w")
print("# n\tLowerbound\t3*Lowerbound\tciLow\tave\tciHigh\tRelErr", file=datafile)

for n in bin.get_inputs():
    print("Running for n=", n)
    out_run = bin.run(n)
    print(*out_run, sep="\t", file=datafile)
datafile.close()

end = time.time()

print("<<<SIMULATION DONE IN {:.4f}s>>>".format(end-start))