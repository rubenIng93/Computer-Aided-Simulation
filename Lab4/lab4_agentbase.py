import random
import numpy as np
from scipy.stats import t

# function that check whether there are otr not infected, exit loop condition
def check_infected():
    if len(set(infected_people == np.zeros(population))) == 1: # there are not infected
        return True 
    else:
        return False

# function that preocedes with the counter of infection days
def update_infected():
    for i in range(len(infected_people)):
        #rnd_inf = round(random.uniform(12, 16))
        if infected_people[i] == infection_duration+1:
            infected_people[i] = 0 # reset the days counter
            recovered_people[i] = 1 # set him as recovered

        elif infected_people[i] > 0 and infected_people[i] < infection_duration+1:
            infected_people[i] += 1
       

# function that check the possibility to meet person complaiant with the contact per day parameter
def update_checklist():
    for i in range(population):
        #rnd_meet = int(random.expovariate(contact_per_day)) # pick a random guy
        #rnd_meet = round(random.uniform(4, 6))
        if check_list[i] > 0 and check_list[i] < 5: # with 5 is now deterministic
            check_list[i] += 1
            #if check_list[i] == 5:
            #    check_list[i] = 0
        elif check_list[i] == 5:
            check_list[i] = 0 # now the person can meet again others individuals


# useful variables
population = 10000
contact_per_day = 0.2 
infection_duration = 14 # days
runs = 10
seed = 22
random.seed(seed)
confidence_level = 0.95

# data structures
runs_S = []
runs_I = []
runs_R = []

subsceptible_people = np.ones(population, dtype=bool) # 0 or 1 starting all subsceptible
infected_people = np.zeros(population) # count the days of infections
recovered_people = np.zeros(population, dtype=bool) # recovered 1 or not 0
i_t = [] # infection by day
s_t = [] # subsceptible by day
r_t = [] # recovered by day

# INITIALIZATION
first_infected = random.randint(0, population-1) # extract the first infected
infected_people[first_infected] = 1 # set the first day
subsceptible_people[first_infected] = 0
i_t.append(1)
s_t.append(population -1)
r_t.append(0)

day = 0

check_list = np.zeros(population) # to check wheter a met b, than a and b
# cannot meet anyone for 5 days

# spread loop
while check_infected() != True: # while there is an infected    

    # update the check list and infected one
    if day != 0:
        update_checklist()
        update_infected() # update recoverd and infected list

    for people in range(len(subsceptible_people)):

        if check_list[people] == 0:
            # in this loop only in the people can meet other persons            

            # with probability 0.2 he meets another person
            #if random.random() < contact_per_day/2:
            # extract the people met
            met = random.randint(0, population-1)
            # check if the one of the two is infected
            if check_list[met] == 0:

                if infected_people[people] > 0:
                    # the first is infected, then becomes also the second
                    # if is in the subsceptible list
                    if subsceptible_people[met]:
                        infected_people[met] = 1 # set the first day of infection
                        subsceptible_people[met] = 0 # not anymore subsceptible

                elif infected_people[met] > 0:
                    if subsceptible_people[people]:
                        infected_people[people] = 1
                        subsceptible_people[people] = 0 # same for the second

                check_list[people] += 1
                check_list[met] += 1
        
    # update lists and day   
    
    actual_subsceptible = np.sum(subsceptible_people)
    actual_recovered = np.sum(recovered_people) # new recovered value
    actual_infected = population - np.unique(infected_people, return_counts=True)[1][0] # count how many infected

    s_t.append(actual_subsceptible)
    r_t.append(actual_recovered)
    i_t.append(actual_infected)

    day += 1
    
    
#print(i_t)
#print(s_t)
#print(r_t)
print(f'The infection lasts {day} days')
print(f'Max infected = {max(i_t)} at day {np.argmax(i_t)}')

# save in a file
datafile = open('Lab4/SIRmodel_agentBased.dat', 'w')
print('day\tS_t\tI_t\tR_t\tcilow\tcih', file=datafile)

for i in range(len(i_t)):
    print(
        i, # the day
        s_t[i], # s_t
        i_t[i], # i_t
        r_t[i], # r_t
        sep='\t',
        file=datafile
    )
datafile.close()