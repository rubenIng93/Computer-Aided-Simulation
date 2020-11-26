import numpy as np
import random
import time
from scipy.stats import t

# initial settings
seed = 2502
confidence_level = 0.95
runs = 5
debug = False
load_balancing = None

class BinNBalls_simulator:
    def __init__(self, runs, seed, confidence_level, debug, load_balancing):
        self.input_list = []
        self.runs = runs
        self.confidence_level = confidence_level
        self.debug = debug
        self.load_balancing = load_balancing

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

    def run(self, n): # n is the number of bins
        random.seed(a=seed)
        maxvec = np.full(self.runs, 0)
        for run in range(self.runs):
            bins = np.full(n, 0)
            for _ in range(n):
                if self.load_balancing is not None:
                    # do the load balancing problem with coefficient = load_balancing
                    # pick at random load_balancing bins
                    args = random.sample(range(0, n-1), self.load_balancing) #retrieve random positions of bins
                    # select and feed the least occupied
                    arg_min_bin = np.argmin(bins[args])
                    bins[args[arg_min_bin]] +=1
                else:
                    bins[random.randint(0, n-1)] += 1
            maxvec[run] = bins.max()
        ave, ci, rel_err = self.evaluate_conficence_interval(maxvec)

        if self.load_balancing is not None:
            theoretical = np.log(np.log(n)) / np.log(self.load_balancing)
            return n, theoretical, ave - ci, ave, ave + ci, rel_err
        else:    
            lower_bound = np.log(n) / np.log(np.log(n))
            return n, lower_bound, 3 * lower_bound, ave - ci, ave, ave + ci, rel_err



bin = BinNBalls_simulator(runs, seed, confidence_level, debug, load_balancing)
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

if load_balancing is not None:
    modality = "_load_balancing"+str(load_balancing)
else:
    modality = ""

datafile = open("Lab1/binsballs"+str(runs)+"runs"+modality+".dat", "w")
if load_balancing is None:
    print("# n\tLowerbound\t3*Lowerbound\tciLow\tave\tciHigh\tRelErr", file=datafile)
else:
    print("# n\tTheoretical\tciLow\tave\tciHigh\tRelErr", file=datafile)


for n in bin.get_inputs():
    print("Running for n=", n)
    out_run = bin.run(n)
    print(*out_run, sep="\t", file=datafile)
datafile.close()

end = time.time()

print("<<<SIMULATION DONE IN {:.4f}s>>>".format(end-start))
