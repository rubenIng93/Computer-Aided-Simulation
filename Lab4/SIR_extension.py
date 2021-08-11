import random
import numpy as np
from scipy.stats import t
import argparse

### setting up the parsers for different studies
parser = argparse.ArgumentParser()
parser.add_argument('--sirs', type=int, help='0 to dont use sirs, 1 instead')
parser.add_argument('--start_lock', type=int, default=0, help='Starting day for the lockdown')
parser.add_argument('--end_lock', type=int, default=0, help='Ending day for the lockdown')
parser.add_argument('--ct_lock', type=float, default=0.02, help='Probability of contact during the lockdown')

args = parser.parse_args()

# sanity check
assert(args.start_lock >= 0 and args.end_lock <= 365)
SIRS = args.sirs
LOCKDOWN = (args.start_lock, args.end_lock)
PROB_CT_LOCK = args.ct_lock


# starting parameters
population = 10000
contact_per_day = 0.2 
infection_duration = 14 # days
runs = 15
seed = 5
random.seed(seed)
np.random.seed(seed)
confidence_level = 0.95
debug = False

'''
Extensions:
-   Probability of being infected not = 1, but extracted from a uniform
    Assumption: 
        * Probability of get infection without mask = 0.7
        * Probability of get infection without mask = 0.15 # not meaningfull results, always degenerative
-   After the recovery, an individual becomes again subsceptible
-   Lockdown period
'''

def evaluate_conficence_interval(x, runs):

    '''
    Compute the confidence interval according to the t of Student

    Parameters:
    ---
    x: narray
        numpy array where the components are values for each run
    runs: int
        how many runs have been launched (to avoid degenerative runs)

    Returns:
    ---
    ave: float
        the average of the vector
    ci: float
        the semi-width of the confidence interval
    rel_err: float
        the relative error
    '''
    
    t_treshold = t.ppf((confidence_level + 1) / 2, df= runs-1)
    ave = x.mean()
    stddev = x.std(ddof=1)
    ci = t_treshold * stddev /np.sqrt(runs)
    if ave != 0:
        rel_err = ci/ ave
    else:
        rel_err = 0
    
    return (ave, ci, rel_err)

def confidence_interval_global(lists, runs):
    '''
    Compute the confidence interval for a list of list

    Parameters: 
    ---
    lists: list of list
        the list of list for each run of the simulation of S or R or I

    runs: int
        how many runs have been launched (to avoid degenerative runs)


    Returns: 
    ---
    result: list of tuples
        a vector with means, ci and relative error for each
        [(ci, mean, rel_err), ...]
    '''
    result = []
    # built an array for each step i.e. for first step all position 0 
    for day in range(len(lists[0])): # is always 365 by construction
        temp_list = []
        for _list in lists:
            temp_list.append(_list[day])
        result.append(evaluate_conficence_interval(np.array(temp_list), runs))

    return result

# class for each individuals
class Person:

    '''
    Class which represents a single individual
    
    Parameters:
    ---
    infection_duration: int
        the average time of the infection duration in days

    sirs: bool
        whether or not to use SIRS model
    '''

    def __init__(self, infection_duration, sirs):
        self.subsceptible = True 
        self.infected = False
        self.recovered = False
        self.sirs = sirs
        self.infection_duration = infection_duration
        if self.sirs:
            self.start_infection = [] # the day
            self.end_infection = [] # the day
        else:
            self.start_infection = -1 # the day
            self.end_infection = -1 # the day
        self.num_infected = 0
        self.day_for_subsceptible = -1
        self.num_infections = 0 # count how many times he get infected
        self.infected_by_day = np.zeros(365, dtype=int)

    def get_infection(self):
        self.infected = True
        self.subsceptible = False
        self.day_for_recovery = np.random.geometric(1/self.infection_duration) +1
        self.infection_time = self.day_for_recovery
        self.num_infections += 1
    
    def get_rid_infection(self):
        self.infected = False
        self.recovered = True
        if self.sirs:
            # modl the temporary immunity [0, 10]
            self.day_for_subsceptible = int(np.random.uniform() * 10)


class AgentBasedModel:

    '''
    Class that represents the world

    Parameters:
    ---
    population: int
        the number of individuals

    beta: float
        contact rate per capita in days^-1
    
    infection_duration: int
        the average time of the infection duration in days

    mask: bool
        whether or not the population wears the mask 

    sirs: bool
        whether or not use the SIRS model

    lockdown: tuple (start, end) (int, int)
        start day and end day for the lockdown; default (0, 0) no lockdown

    ct_prob_lock: float
        probability of contact in case of lockdown

    '''

    def __init__(self, population, contact_per_day, infection_duration, mask, sirs, \
            ct_prob_lock, lockdown=(0, 0)):
        self.population = population
        self.contact_per_day = contact_per_day
        self.infection_duration = infection_duration
        ### Extension variables
        self.contact_in_lockdown = ct_prob_lock
        self.mask = mask
        self.sirs = sirs
        assert(lockdown[0] >= 0 and lockdown[1] <= 365)
        self.lockdown = lockdown
        self.infection_prob_with_mask = 0.15
        self.infection_prob_without_mask = 0.7
        ### Boolean arrays to easily compute statistics exploiting numpy's methods
        self.subsceptibles = np.ones(population, dtype=bool)
        self.infected = np.zeros_like(self.subsceptibles, dtype=bool)
        self.recovered = np.zeros_like(self.subsceptibles, dtype=bool)
        # dict that keeps all the individuals
        self.person_dict = {k:Person(infection_duration, self.sirs) for k in range(population)}
        # lists aimed to track the evolution of the parameters according to days
        self.s_t = []
        self.i_t = []
        self.r_t = []
        self.Rt = [] # rt parameter 
        

    def start_disease(self):
        '''
        Pick at random the first infected and update its own status
        '''
        first_infected_idx = random.randint(0, self.population - 1)
        #first_infected_idx = 0
        self.person_dict[first_infected_idx].get_infection()
        if not self.sirs:
            self.person_dict[first_infected_idx].start_infection = 0
        else:
            self.person_dict[first_infected_idx].start_infection.append(0)
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
            if p.infected:
                p.day_for_recovery -= 1
                if p.day_for_recovery == 0:
                    p.get_rid_infection()
                    self.recovered[k] = 1
                    self.infected[k] = 0
                    if not self.sirs:
                        p.end_infection = day
                    else:
                        p.end_infection.append(day)
                # set automatically to 365 the end day if it's day 365
                elif day > 364 and self.sirs:
                    p.end_infection.append(day)
            if p.day_for_subsceptible >= 0: # temporary immunity
                p.day_for_subsceptible -= 1
                if p.day_for_subsceptible < 0:
                    p.subsceptible = True
                    p.recovered = False
                    self.subsceptibles[k] = 1
                    self.recovered[k] = 0

    def update_trackers(self):

        '''
        Register in the data structures the new values for each parameters
        '''

        actual_subsceptible = np.sum(self.subsceptibles)
        actual_recovered = np.sum(self.recovered) # new recovered value
        actual_infected = np.sum(self.infected) # count how many infected
        self.s_t.append(actual_subsceptible)
        self.r_t.append(actual_recovered)
        self.i_t.append(actual_infected)

    def compute_rt(self):

        '''
        Compute the Rt defined as:
        "average number of secondary infections produced by a typical infective
        during the entire period of infectiouness at time t".
        '''

        rt = []
        for day in range(366):
            tot_infected_today = 0
            tot_infectors_today = 0
            for person in self.person_dict.values():

                ### SIRS case
                if self.sirs:
                    if person.num_infections > 0 :
                        # get the end of the first infection
                        current_end = person.end_infection[-person.num_infections]
                        current_start = person.start_infection[-person.num_infections]
                        tot_infections = len(person.end_infection)

                        if tot_infections >0 and current_end < day:
                            # descrease the counter of the infections 
                            person.num_infections -= 1

                        if day >= current_start and day <= current_end:                    
                            
                            # slice the infection list of each person according to the day no sirs
                            infected_by_current_day = person.infected_by_day[day:current_end]
                            
                            num_infected_from_today = np.sum(infected_by_current_day)
                            #print('FROM today: ',num_infected_from_today)

                            tot_infected_today += num_infected_from_today
                            tot_infectors_today += 1

                else:
                    ### SIR case
                    if day >= person.start_infection and day <= person.end_infection \
                        and not person.subsceptible:
                        
                        # slice the infection list of each person according to the day
                        infected_by_current_day = person.infected_by_day[day:]
                        # get all the infected from the current day onwards
                        num_infected_from_today = np.sum(infected_by_current_day)

                        tot_infected_today += num_infected_from_today
                        tot_infectors_today += 1
            if tot_infectors_today == 0:
                rt.append(0)
            else:
                rt.append(tot_infected_today/tot_infectors_today)
        return rt


    def simulate(self, debug=False):

        '''
        Go through the year updating infections and recovery

        Parameters:
        ---
        
        debug: bool, default False
            whether of not to debug
        
        '''
        
        day = 0
        self.start_disease()
        # loop over the year

        while day < 365:            
            
            # check the metrics
            if day % 10 == 0 and debug:
                print(f'Subsceptible: {np.sum(self.subsceptibles)}')
                print(f'Infected: {np.sum(self.infected)}')
                print(f'Recovered: {np.sum(self.recovered)}')
                print(f'Day: {day}')

            lockdown = False
            # check if the lockdown is planned
            if self.lockdown[1]:
                
                if day >= self.lockdown[0] and day <= self.lockdown[1] :
                    # save old beta and reduce the current
                    lockdown = True

            
            # indexes of infected people
            infected_idxs = np.argwhere(self.infected==1).flatten()
            # loop over the infected
            for infected_id in infected_idxs:
                # get the person
                infected = self.person_dict[infected_id]
                # draw from poisson how many individuals the infected meets
                if not lockdown:
                    num_met_today = np.random.poisson(self.contact_per_day) 
                else:
                    num_met_today = np.random.poisson(self.contact_in_lockdown) 

                num_infected_today = 0

                if num_met_today > 0:
                    # sample the indexes of met individuals
                    met_idxs = random.sample(range(self.population), num_met_today)
                    for idx in met_idxs:
                        # get the person
                        met = self.person_dict[idx]
                        
                        # register the infection in met is subsceptible
                        if met.subsceptible:
                            infect = False
                            if self.mask:
                                # draw the probability to be infected
                                if self.infection_prob_with_mask >= np.random.uniform():
                                    infect = True
                            else:
                                # draw the probability to be infected
                                if self.infection_prob_without_mask >= np.random.uniform():
                                    infect = True
                            if infect:
                                met.get_infection()
                                self.infected[idx] = 1
                                self.subsceptibles[idx] = 0
                                if not self.sirs:
                                    met.start_infection = day
                                else:
                                    met.start_infection.append(day)
                                num_infected_today += 1
                # register how many person he infected in the day   
                infected.infected_by_day[day] += num_infected_today


            # update all the parameters
            self.update_trackers()                     

            day += 1

            # update the parameters according to the new day
            self.new_day(day)


        time = 365 if self.i_t[-1] > 0 else np.argmin(self.i_t)
        print(f'The infection lasted {time} day')
        print(f'With a peak of {np.max(self.i_t)} infected at day {np.argmax(self.i_t)}')



# data structures to keep all the runs
runs_s = [] # list of s_t for each run
runs_i = []
runs_r = []
Rt_dict = {} # rt for each run

print(5*'#' + ' Initial setting ' + 5*'#')
print(f'Population: {population}    seed: {seed}')
print(f'c. level:   {confidence_level}')
print(f'SIRS: {"true" if SIRS == 1 else "false"}')
print(f'LOCKDOWN:   {LOCKDOWN}')
print(f'BETA in Lockdown:   {PROB_CT_LOCK if LOCKDOWN[1] > 0 else "--"}')


# data structures to outputs some useful statistics
infection_peaks = []
infection_peak_days = []
infection_time = []

#### loop over the number of runs
for run in range(runs):

    print(f'\nSimulate run {run}')

    simulator = AgentBasedModel(population, contact_per_day, infection_duration, \
            mask=False, sirs=SIRS, lockdown=LOCKDOWN, ct_prob_lock=PROB_CT_LOCK)
    simulator.simulate()

    # avoid degenerative runs
    if np.max(simulator.i_t) < 10:
        print('Degenerative run detected')
    else:
        # save all
        runs_s.append(simulator.s_t)
        runs_i.append(simulator.i_t)
        runs_r.append(simulator.r_t)
        Rt_dict[run] = simulator.compute_rt()
        ###
        infection_peaks.append(np.max(simulator.i_t))
        infection_peak_days.append(np.argmax(simulator.i_t))
        if simulator.i_t[-1] > 0:
            infection_time.append(365)
        else:
            infection_time.append(np.argmin(simulator.i_t))      
    
### Compute and print interesting outputs
avg_peak, ci_peak, _ = evaluate_conficence_interval(np.array(infection_peaks), len(infection_peaks))
avg_peak_d, ci_peak_d, _ = evaluate_conficence_interval(np.array(infection_peak_days), len(infection_peak_days))
avg_time, ci_time, _ = evaluate_conficence_interval(np.array(infection_time), len(infection_time))

print(40*'#')
print(f'{len(runs_s)} runs, {runs - len(runs_s)} discarded since degeneratives\n')
print('Average values:')
print(f'Peak of Infection:  {avg_peak:.4f}')
print(f'Peak Day:           {avg_peak_d:.4f}')
print(f'Duration:           {avg_time:.4f}\n')
print('Confidence Intervals:')
# check non negativity
cil_peak = avg_peak - ci_peak
if cil_peak < 0:
    cil_peak = 0
cil_day = avg_peak_d - ci_peak_d
if cil_day < 0:
    cil_day = 0
cil_time = avg_time - ci_time
if cil_time < 0:
    cil_time = 0
# not over 365 days
cih = avg_time + ci_time
print(f'Peak of Infection:  [ {avg_peak - ci_peak:.4f} ; {avg_peak + ci_peak:.4f} ]')
print(f'Peak Day:           [ {avg_peak_d - ci_peak_d:.4f} ; {avg_peak_d + ci_peak_d:.4f} ]')
print(f'Duration:           [ {avg_time - ci_time:.4f} ; {cih if cih <= 365 else "+"} ]')


# get the cis
result_s = confidence_interval_global(runs_s, len(runs_s))
result_i = confidence_interval_global(runs_i, len(runs_s))
result_r = confidence_interval_global(runs_r, len(runs_s))
rt_ci = confidence_interval_global(list(Rt_dict.values()), len(runs_s))

## define the filename
filename = ''
if SIRS:
    if LOCKDOWN[1]:
        filename = 'simulative_SIRS_lock.dat'
    else:
        filename = 'simulative_SIRS.dat'
else:
    if LOCKDOWN[1]:
        filename = 'SIR_ext_lock.dat'
    else:
        filename = 'SIR_ext.dat'

#### save in a datafile the metrics
datafile = open(f'Lab4/{filename}', 'w')
print('day\tmean(S_t)\tciLow(S_t)\tciHigh(S_t)\trelerr(S_t)' + \
    '\tmean(I_t)\tciLow(I_t)\tciHigh(I_t)\trelerr(I_t)' + \
        '\tmean(R_t)\tciLow(R_t)\tciHigh(R_t)\trelerr(R_t)' +\
            '\tmean(RT)\tciLow(RT)\tciHigh(RT)\trelerr(RT)', file=datafile)

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
        rt_ci[i][0], # s_t mean
        rt_ci[i][0] - rt_ci[i][1], # cilow
        rt_ci[i][0] + rt_ci[i][1], # cihigh
        rt_ci[i][2], # rel error
        sep='\t',
        file=datafile
    )
datafile.close()
