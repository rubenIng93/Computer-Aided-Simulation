import numpy as np
import random
import math
from scipy.stats import t
import time
import os


# initial parameters
runs = 1000
seed = 22
#m_people = 23
n_elements = 365
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

    def evaluate_conficence_interval(self, x): # a vector
        t_treshold = t.ppf((self.ci_level + 1) / 2, df= self.runs-1)
        ave = x.mean()
        stddev = x.std(ddof=1)
        ci = t_treshold * stddev /np.sqrt(self.runs)
        rel_err = ci/ ave
        #f self.debug:
            #print("Min: {:.2f}, Ave: {:.2f}, Max: {:.2f}".format(x.min(), ave, x.max()))
        return ave, ci, rel_err

    def run(self, m_people, _type='prob'):
        random.seed(a=self.seed)
        # initialize the vector of n element with all 0
        conflict_per_run = np.full(self.runs, 0)
        # vector that computes the whether there's a conflict
        # for the number m of people
        num_people_conflict = np.full(self.runs, 0)    

        for run in range(self.runs):
            chosen_element = np.full(self.n_elements, 0) # array that check the conflict in a single run
            for m in range(m_people): # parameter given in the function
                rnd_element = random.randint(0, self.n_elements-1)
                if chosen_element[rnd_element] != 0: # if the place rnd is occupied
                    conflict_per_run[run] = 1 # warn a conflict
                    if num_people_conflict[run] == 0:
                        # it triggers only the first time -> takes only the minimum
                        num_people_conflict[run] = m
                        #print(f'Conflict in run {run} with {m}')
                    break
                else:
                    chosen_element[rnd_element] = 1

        # return the probability of conflicts for the given setting
        if _type == 'prob': 
            p_conflicts = float(np.sum(conflict_per_run) / self.runs) # it is also the mean
            std_dev = math.sqrt(p_conflicts * (1 - p_conflicts)) # standar deviation
            theoretical_probability = 1 - math.exp(-(m_people**2 / (2 * self.n_elements)))
            ci, rel_err = self.retrieve_ci(p_conflicts, std_dev) # compute the ci
            ciLow = p_conflicts - ci
            ciHigh = p_conflicts + ci

            # avoid to return probabilities > 1 or < 0
            if ciHigh > 1.0:
                ciHigh = 1.0
            if ciLow <= 0.0:
                ciLow = 0.0

            return m_people, theoretical_probability, p_conflicts,\
                    ciLow, ciHigh, rel_err
        
        elif _type == 'minimum':
            # minimum m_people for conflicts part
            min_m_mean, min_ci, min_m_rel_err = self.evaluate_conficence_interval(num_people_conflict) # retrieve the ci
            min_m_cihigh = min_m_mean + min_ci
            min_m_cilow = min_m_mean - min_ci
            # theoretical value
            theo = math.sqrt(math.pi / 2 * self.n_elements)
            
            return self.n_elements, min_m_cilow, min_m_mean, min_m_cihigh, min_m_rel_err, theo

        
        
# make a folder
if os.listdir(os.getcwd()+"/Lab2").__contains__("data") == False:
    os.mkdir(os.getcwd()+"/Lab2/data")        
 

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
datafile = open("Lab2/data/birthdayparadox"+str(n_elements)+"elements"+str(runs)+"runs.dat", "w")
print("# peoples\tTheoretical\tSimulations\tciLow\tciHigh\tRelErr", file=datafile)

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

# loop for print the file

for m in range(1, stop_value, step):
    print("Running for m=", m)
    out_run = sim.run(m)
    print(*out_run, sep="\t", file=datafile)
datafile.close()

# retrieve the minimum m for looking a conflict, just printed
n, cilow, avg, cihigh, err, theo = sim.run(stop_value, _type='minimum')
print(f'For {n} elements are needed on avg {avg} people to get a conflict')
print(f'Whose 95% CI is [{cilow:.3f}, {cihigh:.3f}] - Theorical = {theo:.3f}')

end = time.time()

print("<<<SIMULATION DONE IN {:.4f}s>>>".format(end-start))
