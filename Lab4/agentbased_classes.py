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

        print(f'The infection lasted {day - 1} day')
        print(f'With a peak of {np.max(self.i_t)} infected at day {np.argmax(self.i_t)}')

        
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


    
simulator = AgentBasedModel(population, contact_per_day, infection_duration)
simulator.simulate()
simulator.generate_file()