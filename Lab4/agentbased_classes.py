import random
import numpy as np
from scipy.stats import t

# starting parameters
population = 10000
contact_per_day = 0.2 
infection_duration = 14 # days
runs = 10
seed = 100
random.seed(seed)
confidence_level = 0.95
debug = False

def evaluate_conficence_interval(x):
        t_treshold = t.ppf((confidence_level + 1) / 2, df= runs-1)
        ave = x.mean()
        stddev = x.std(ddof=1)
        ci = t_treshold * stddev /np.sqrt(runs)
        rel_err = ci/ ave
        
        return (ave, ci, rel_err)

def confidence_interval_global(lists):
    '''
    Args: the list of list for each run of the simulation of S or R or I
    Returns: a vector with means, ci and relative error for each
    [(ci, mean, rel_err), ...]
    '''
    result = []
    # built an array for each step i.e. for first step all position 0 
    for day in range(len(lists[0])): # is always 365 by construction
        temp_list = []
        for _list in lists:
            temp_list.append(_list[day])
        result.append(evaluate_conficence_interval(np.array(temp_list)))

    return result

# class for each individuals
class Person:

    def __init__(self, infection_duration):
        self.subsceptible = True 
        self.infected = False
        self.recovered = False
        self.last_contact = 0
        self.infection_duration = infection_duration

    def get_infection(self):
        self.infected = True
        self.subsceptible = False
        self.day_for_recovery = self.infection_duration
    
    def get_rid_infection(self):
        self.infected = False
        self.recovered = True


class AgentBasedModel:

    def __init__(self, population, contact_per_day, infection_duration):
        self.population = population
        self.contact_per_day = contact_per_day
        self.infection_duration = infection_duration
        self.subsceptibles = np.ones(population, dtype=bool)
        self.infected = np.zeros_like(self.subsceptibles, dtype=bool)
        self.recovered = np.zeros_like(self.subsceptibles, dtype=bool)
        self.person_dict = {k:Person(infection_duration) for k in range(population)}
        self.s_t = []
        self.i_t = []
        self.r_t = []
        

    def start_disease(self):
        '''
        Pick at random the first infected and update its own status
        '''
        first_infected_idx = random.randint(0, self.population - 1)
        self.person_dict[first_infected_idx].get_infection()
        self.infected[first_infected_idx] = 1
        self.subsceptibles[first_infected_idx] = 0
        self.i_t.append(1)
        self.s_t.append(self.population - 1)
        self.r_t.append(0)
    
    def new_day(self):
        '''
        Update status according to the new day
        i.e. decrease both infection_duration and last contact
        '''
        
        for k, p in enumerate(self.person_dict.values()):
            if p.last_contact > 0:
                p.last_contact -= 1
            if p.infected:
                p.day_for_recovery -= 1
                if p.day_for_recovery == 0:
                    p.get_rid_infection()
                    self.recovered[k] = 1
                    self.infected[k] = 0

    def update_trackers(self):

        actual_subsceptible = np.sum(self.subsceptibles)
        actual_recovered = np.sum(self.recovered) # new recovered value
        actual_infected = self.population - np.unique(self.infected, return_counts=True)[1][0] # count how many infected

        self.s_t.append(actual_subsceptible)
        self.r_t.append(actual_recovered)
        self.i_t.append(actual_infected)

    def simulate(self):
        
        day = 0
        self.start_disease()
        while day < 365:
            # check the metrics
            if day % 1000 == 0 and debug:
                print(f'Subsceptible: {np.sum(self.subsceptibles)}')
                print(f'Infected: {np.sum(self.infected)}')
                print(f'Recovered: {np.sum(self.recovered)}')
                print(f'Day: {day}')

            for k, p in enumerate(self.person_dict.values()):

                if p.last_contact == 0:
                    # he can meet someone
                    met_idx = random.randint(0, population-1)
                    met = self.person_dict[met_idx]
                    if met.last_contact == 0:
                        # update the meeting counter
                        met.last_contact = 1 / self.contact_per_day -1 # a person every 5 days
                        p.last_contact = 1 / self.contact_per_day -1
                        # check whether the 2 persons are infected
                        if met.infected and p.subsceptible:
                            # infect the other person
                            p.get_infection()
                            self.subsceptibles[k] = 0
                            self.infected[k] = 1
                        elif p.infected and met.subsceptible:
                            met.get_infection()
                            self.subsceptibles[met_idx] = 0
                            self.infected[met_idx] = 1
            
            # update all the parameters
            self.new_day()
            self.update_trackers()

            day += 1

        #print(f'The infection lasted {day - 1} day')
        #print(f'With a peak of {np.max(self.i_t)} infected at day {np.argmax(self.i_t)}')    


    '''not needed anymore'''
    def generate_file(self):

        datafile = open('Lab4/SIRmodel_agentBased2.dat', 'w')
        print('day\tS_t\tI_t\tR_t\tcilow\tcih', file=datafile)

        for i in range(len(self.i_t)):
            print(
                i, # the day
                self.s_t[i], # s_t
                self.i_t[i], # i_t
                self.r_t[i], # r_t
                sep='\t',
                file=datafile
            )
        datafile.close()


# data structures to keep all the runs
runs_s = [] # list of s_t for each run
runs_i = []
runs_r = []

print(5*'#' + ' Initial setting ' + 5*'#')
print(f'Population: {population}    seed: {seed}')
print(f'c. level:   {confidence_level}')


for run in range(runs):

    print(f'\nSimulate run {run}')

    simulator = AgentBasedModel(population, contact_per_day, infection_duration)
    simulator.simulate()

    # save all
    runs_s.append(simulator.s_t)
    runs_i.append(simulator.i_t)
    runs_r.append(simulator.r_t)

# get the cis
result_s = confidence_interval_global(runs_s)
result_i = confidence_interval_global(runs_i)
result_r = confidence_interval_global(runs_r)


datafile = open('Lab4/provaCI.dat', 'w')
print('day\tmean(S_t)\tciLow(S_t)\tciHigh(S_t)\trelerr(S_t)' + \
    '\tmean(I_t)\tciLow(I_t)\tciHigh(I_t)\trelerr(I_t)' + \
        '\tmean(R_t)\tciLow(R_t)\tciHigh(R_t)\trelerr(R_t)', file=datafile)

for i in range(len(result_s)):
    print(
        i, # the day
        result_s[i][0], # s_t mean
        result_s[i][0] - result_s[i][1], # cilow
        result_s[i][0] + result_s[i][1], # cihigh
        result_s[i][2], # rel error
        result_i[i][0], # s_t mean
        result_i[i][0] - result_i[i][1], # cilow
        result_i[i][0] + result_i[i][1], # cihigh
        result_i[i][2], # rel error
        result_r[i][0], # s_t mean
        result_r[i][0] - result_r[i][1], # cilow
        result_r[i][0] + result_r[i][1], # cihigh
        result_r[i][2], # rel error
        sep='\t',
        file=datafile
    )
datafile.close()

