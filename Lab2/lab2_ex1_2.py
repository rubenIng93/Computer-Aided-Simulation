import numpy as np
import random
import math
from scipy.stats import t
import time


# initial parameters
runs = 1000
seed = 22
#m_people = 23
n_elements = 10**6
ci_level = 0.95

class BirthdayParadoxSimulator:
    def __init__(self, n_elements, runs, seed, ci_level):
        #self.m_people = m_people
        self.n_elements = n_elements
        self.runs = runs
        self.seed = seed
        self.ci_level = ci_level

    def retrieve_ci(self, mean, stddev):
        t_treshold = t.ppf((self.ci_level + 1) / 2, df= self.runs-1)
        ci = t_treshold * stddev /np.sqrt(self.runs)
        #rel_err = ci / mean
        rel_err = 0 # must be restored
        return ci, rel_err

    def run(self, m_people):
        random.seed(a=self.seed)
        # initialize the vector of n element with all 0
        conflict_per_run = np.full(self.runs, 0)
        # vector that computes the whether there's a conflict
        # for the number m of people
        num_people_conflict = np.full(self.runs, 0)    

        for run in range(self.runs):
            chosen_element = np.full(self.n_elements, 0)
            for m in range(m_people):
                rnd_element = random.randint(0, self.n_elements-1)
                if chosen_element[rnd_element] != 0:
                    conflict_per_run[run] = 1
                    if num_people_conflict[run] == 0:
                        # it triggers only the first time -> takes only the minimum
                        num_people_conflict[run] = m
                    break
                else:
                    chosen_element[rnd_element] = 1
        # return the probability of conflicts for the given setting
        p_conflicts = float(np.sum(conflict_per_run) / self.runs) # it is also the mean
        std_dev = math.sqrt(p_conflicts * (1 - p_conflicts)) # standar deviation
        theoretical_probability = 1 - math.exp(-(m_people**2 / (2 * self.n_elements)))
        ci, rel_err = self.retrieve_ci(p_conflicts, std_dev)
        ciLow = p_conflicts - ci
        ciHigh = p_conflicts + ci

        # minimum m_people for conflicts part
        min_m_mean = np.sum(num_people_conflict) / self.runs
        min_m_stddev = np.std(num_people_conflict)
        min_m_ci, min_m_rel_err = self.retrieve_ci(min_m_mean, min_m_stddev)
        min_m_cilow = min_m_mean - min_m_ci
        min_m_cihigh = min_m_mean + min_m_ci

        # avoid to return probabilities > 1 or < 0
        if ciHigh > 1.0:
            ciHigh = 1.0
        if ciLow <= 0.0:
            ciLow = 0.0
        return m_people, theoretical_probability, p_conflicts,\
                    ciLow, ciHigh, rel_err, min_m_cilow, min_m_mean, \
                        min_m_cihigh, min_m_rel_err
  

print("***INITIAL SETTINGS***")
print("Birthday Paradox simulation for {} elements".format(n_elements))
print("Initial seed: ", seed)
print("Confidence level: ", ci_level)
print("Number of runs: ", runs)
print("***END INITIAL SETTINGS***\n")

start = time.time()
print("<<<START SIMULATION>>>")

sim = BirthdayParadoxSimulator(n_elements, runs, seed, ci_level)
#print("Probability of conflicts vs theoretical: ", sim.run(m_people))
datafile = open("Lab2/birthdayparadox"+str(n_elements)+"elements"+str(runs)+"runs.dat", "w")
print("# peoples\tTheoretical\tSimulations\tciLow\tciHigh\tRelErr\tciLowPeople\tmeanPeople\tciHighPeople\tpeopleRelErr", file=datafile)

# defining the step since with more elements the simulator slows down a lot
if n_elements == 365:
    step = 1
elif n_elements == 10**5:
    if runs == 100:
        step = 10
    else:
        step = 100
else:
    step = 100

stop_value = int(3.5*math.sqrt(n_elements))
#stop_value = n_elements


for m in range(1, stop_value, step):
    print("Running for m=", m)
    out_run = sim.run(m)
    print(*out_run, sep="\t", file=datafile)
datafile.close()
            
end = time.time()

print("<<<SIMULATION DONE IN {:.4f}s>>>".format(end-start))
