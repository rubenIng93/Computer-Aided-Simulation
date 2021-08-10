import numpy as np
import random
from scipy.stats import t

population = 10000
beta = 0.2 # contact per day, per capita
gamma = 1/14 # recovery rate
seed = 700
confidence_level = 0.95
runs = 10


# set the seed
np.random.seed(seed)
random.seed(seed)


def evaluate_conficence_interval(x, runs):

    '''
    Compute the confidence interval according to the t of Student

    Parameters:
    ---
    x: narray
        numpy array where the components are values for each run

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


class Person:

    def __init__(self, gamma):
        '''
        Class which represents a single individual
        
        Parameters:
        ---
        gamma: float
            the rate of recovery in [days^-1]
        '''
        self.subsceptible = True
        self.infected = False
        self.recovered = False
        assert(isinstance(gamma, float))
        self.gamma = gamma
        self.last_contact = -1
        self.num_contact = 0
        

    def get_infected(self):
        self.subsceptible = False
        self.infected = True
        self.day_to_recovery = self.infected_day = np.random.poisson(lam=1/self.gamma) 


    def get_recovered(self):
        self.infected = False
        self.recovered = True


class World:

    '''
    Class that represents the world

    Parameters:
    ---
    beta: float
        contact rate per capita in days^-1
    
    gamma: float
        recovery rate in days^-1

    population: int
        the number of individuals
    '''

    def __init__(self, beta, gamma, population):
        self.beta = beta
        self.gamma = gamma
        self.population = population

        self.person_dict = {i: Person(self.gamma) for i in range(self.population)}
        self.infected_array = np.zeros(self.population, bool)
        self.subsceptibles_array = np.ones(self.population, bool)
        self.recovered_array = np.zeros(self.population, bool)
        self.S_t = [] # subsceptibles
        self.I_t = [] # infected
        self.R_t = [] # recovered
        # previous lists' length is == tot day (i.e. 365)

    def start_disease(self):
        '''
        Sample the first infected and update its state accordingly
        '''
        first_infected_idx = random.randint(0, self.population -1)
        first_infected = self.person_dict[first_infected_idx]
        first_infected.get_infected()
        self.S_t.append(population-1)
        self.I_t.append(1)
        self.R_t.append(0)
        self.infected_array[first_infected_idx] = 1

    def update_last_contact(self):
        '''
        Descrease the last contact day for each person
        '''
        for person in self.person_dict.values():
            if person.last_contact >= 0:
                person.last_contact -= 1
    

    def simulate_day(self, day, debug=False):
        '''
        Go through a day updating infections and recovery

        Parameters:
        ---
        day: int
            The day where to simulate
        debug: bool, default False
            whether of not to debug
        
        Returns:
        ---
        S_T: int
            daily subscepticles
        I_T: int
            daily infected
        R_T: int
            daily recovered
        '''

        # check if there are infected
        infected_idx = np.argwhere(self.infected_array == 1).flatten()
        subsceptiple_idx = np.argwhere(self.subsceptibles_array == 1).flatten()
        recovered_count = np.sum(self.recovered_array)

        if day % 30 and debug:
            print(f'There are {len(subsceptiple_idx)} subsceptibles at day {day}')
            print(f'There are {len(infected_idx)} infected at day {day}')
            print(f'There are {recovered_count} recovered at day {day}\n')

        # update daily statistics
        s_t = len(subsceptiple_idx)
        i_t = len(infected_idx)
        r_t = recovered_count
        
        # update the couter for recovery (avoid the first day)
        if day > 0:
            for idx in infected_idx:
                infected = self.person_dict[idx]
                infected.day_to_recovery -= 1
                # set it as recovered if the counter goes to zero
                if infected.day_to_recovery == 0:
                    infected.get_recovered()
                    self.infected_array[idx] = 0
                    self.recovered_array[idx] = 1
                    # update daily statistics after recovery
                    i_t -= 1
                    r_t += 1
        
        # update the tracking of last contact
        self.update_last_contact()

        # spread the infection
        for idx in infected_idx:
            # check if he can have a contact
            infected = self.person_dict[idx]
            if infected.last_contact < 0:
                # draw from the distribution in what time he can move
                future_contact = np.random.geometric(self.beta) +1
                infected.last_contact = future_contact
                # sample the index of a person to be infected
                met_idx = random.randint(0, self.population-1)
                met = self.person_dict[met_idx]
                # check if the met can meet someone, otherwise
                # sample another individual
                while met.last_contact >= 0 or met_idx == idx:
                    met_idx = random.randint(0, self.population-1)
                    met = self.person_dict[met_idx]
                # update the last contact of the met
                future_contact = np.random.geometric(self.beta) +1
                met.last_contact = future_contact
                # check if the met can be infected
                infected.num_contact += 1
                if met_idx in subsceptiple_idx:
                    # infect him!
                    met.get_infected()
                    self.subsceptibles_array[met_idx] = 0
                    self.infected_array[met_idx] = 1
                    i_t += 1
                    s_t -= 1
                    
        

        self.S_t.append(s_t)
        self.R_t.append(r_t)
        self.I_t.append(i_t)


# data structures for the runs
runs_s = [] # list of s_t for each run
runs_i = []
runs_r = []

print(5*'#' + ' Initial setting ' + 5*'#')
print(f'Population: {population}    seed: {seed}')
print(f'c. level:   {confidence_level}')


for run in range(runs):

    print(f'\nSimulate run {run}')

    simulator = World(beta, gamma, population)
    simulator.start_disease()

    for day in range(1, 365):
        simulator.simulate_day(day)

    print(5*'#'+'STATS'+5*'#')
    print(f'Max infected:   {np.max(simulator.I_t)}')
    print(f'Reached at day: {np.argmax(simulator.I_t)}\n')  

    tot_beta = 0
    tot_i = 0
    for p in simulator.person_dict.values():
        if not p.subsceptible:
            tot_i += 1
            tot_beta += p.num_contact / p.infected_day

    print(f'Simulative Beta: {tot_beta / tot_i:.4f}\n')

    # avoid degenerative runs
    if np.max(simulator.I_t) < 10:
        print('Degenerative run detected')
    else:
        # save all
        runs_s.append(simulator.S_t)
        runs_i.append(simulator.I_t)
        runs_r.append(simulator.R_t)

# get the cis
result_s = confidence_interval_global(runs_s, len(runs_s))
result_i = confidence_interval_global(runs_i, len(runs_s))
result_r = confidence_interval_global(runs_r, len(runs_s))
#rt_ci = confidence_interval_global(Rt_dict.values())


datafile = open('Lab4/simulative_SIR.dat', 'w')
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







        
    
