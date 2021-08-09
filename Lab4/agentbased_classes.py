import random
import numpy as np
from scipy.stats import t

# starting parameters
population = 10000
contact_per_day = 0.1 
infection_duration = 14 # days
runs = 1
seed = 145
random.seed(seed)
np.random.seed(seed)
confidence_level = 0.95
debug = False

def evaluate_conficence_interval(x):
        t_treshold = t.ppf((confidence_level + 1) / 2, df= runs-1)
        ave = x.mean()
        stddev = x.std(ddof=1)
        ci = t_treshold * stddev /np.sqrt(runs)
        if ave != 0:
            rel_err = ci/ ave
        else:
            rel_err = 0
        
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
        self.start_infection = -1 # the day
        self.end_infection = -1 # the day
        self.infected_by_days = []

    def get_infection(self):
        self.infected = True
        self.subsceptible = False
        self.day_for_recovery = np.random.geometric(1/self.infection_duration)
        self.infection_time = self.day_for_recovery
    
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
        self.Rt = [] # rt parameter 
        

    def start_disease(self):
        '''
        Pick at random the first infected and update its own status
        '''
        first_infected_idx = random.randint(0, self.population - 1)
        self.person_dict[first_infected_idx].get_infection()
        self.person_dict[first_infected_idx].start_infection = 0
        self.infected[first_infected_idx] = 1
        self.subsceptibles[first_infected_idx] = 0
        self.i_t.append(1)
        self.s_t.append(self.population - 1)
        self.r_t.append(0)
        
    
    def new_day(self, day):
        '''
        Update status according to the new day
        i.e. decrease both infection_duration and last contact
        '''
        
        for k, p in enumerate(self.person_dict.values()):
            if p.last_contact >= 0:
                p.last_contact -= 1
            if p.infected:
                p.day_for_recovery -= 1
                if p.day_for_recovery == 0:
                    p.get_rid_infection()
                    self.recovered[k] = 1
                    self.infected[k] = 0
                    p.end_infection = day

    def update_trackers(self):

        actual_subsceptible = np.sum(self.subsceptibles)
        actual_recovered = np.sum(self.recovered) # new recovered value
        actual_infected = np.sum(self.infected) # count how many infected
        self.s_t.append(actual_subsceptible)
        self.r_t.append(actual_recovered)
        self.i_t.append(actual_infected)

    def compute_rt(self):

        rt = []
        for day in range(365):
            tot_i = 0
            tot_p = 0
            for person in self.person_dict.values():
                if day > person.start_infection and day <= person.end_infection:
                    print(f'Duration: {person.infection_time}, list: {len(person.infected_by_days)}')
                    tot_i += person.infected_by_days.pop(0)
                    tot_p += 1
            if tot_p == 0:
                rt.append(0)
            else:
                rt.append(tot_i/tot_p)
        return rt


    def simulate(self):
        
        day = 0
        self.start_disease()
        while day < 365:            
            
            # check the metrics
            if day % 10 == 0 and debug:
                print(f'Subsceptible: {np.sum(self.subsceptibles)}')
                print(f'Infected: {np.sum(self.infected)}')
                print(f'Recovered: {np.sum(self.recovered)}')
                print(f'Day: {day}')
            
             # indexes of infected people
            infected_idxs = np.argwhere(self.infected==1).flatten()
            print(len(infected_idxs))
            for infected_id in infected_idxs:
                infected = self.person_dict[infected_id]
                if infected.last_contact < 0:
                    # draw from poisson the number of contact
                    next_contact = np.random.geometric(self.contact_per_day)
                    # check if there is at least 1 contact

                    # sample the indexes
                    met_index = random.randint(0, population-1)
                    met = self.person_dict[met_index]
                    # verify that is not the same person
                    while infected_id == met_index or met.last_contact <0:
                        met_index = random.randint(0, population-1)
                        met = self.person_dict[met_index]
                        
                        if met.subsceptible:
                            met.get_infection()
                            self.infected[met_index] = 1
                            self.subsceptibles[met_index] = 0
                            #num_infected += 1
                # register how many person he infected in the day            
                #infected.infected_by_days.append(num_infected)
                            


            # update all the parameters
            self.update_trackers()                     

            day += 1

            self.new_day(day)


        #self.Rt = self.compute_rt()

        print(f'The infection lasted {np.argmin(self.i_t) - 1} day')
        print(f'With a peak of {np.max(self.i_t)} infected at day {np.argmax(self.i_t)}')    


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
Rt_dict = {} # rt for each run

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
    Rt_dict[run] = simulator.Rt

# get the cis
result_s = confidence_interval_global(runs_s)
result_i = confidence_interval_global(runs_i)
result_r = confidence_interval_global(runs_r)
#rt_ci = confidence_interval_global(Rt_dict.values())


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

