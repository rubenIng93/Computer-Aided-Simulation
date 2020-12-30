import numpy as np
# Implementation for the SIR model

# fixed parameters
peoples = 10000
contact_per_day = 0.2
infection_period = 14 # in days
debug = True


# indicators at time 0
s_t = peoples - 1 # susceptible peoples
i_t = 1 # infected people
r_t = 0 # removed peoples

gamma = 1 / (infection_period) # recovery rate in days
beta = contact_per_day # transmission rate in days

# structures to track trends
days = 0
s_tracker = []
i_tracker = []
r_tracker = []

# initialize with the first values
s_tracker.append(s_t)
i_tracker.append(i_t)
r_tracker.append(r_t)

# iterate untill the infected peoples are 0 
while int(i_t) > 0:
    
    s_t = s_t - (beta / peoples) * s_tracker[days] * i_tracker[days] 
    i_t = i_t + (beta / peoples) *  s_tracker[days] * i_tracker[days] - gamma * i_tracker[days]
    r_t = r_t + gamma * i_tracker[days]

    days += 1 # new day

    if debug:
        if days % 10 == 0:
            print(f"DAY {days}")
            print(f"SUSCEPTIPLE PEOPLES = {int(s_t)}")
            print(f"INFECTED PEOPLES = {int(i_t)}")
            print(f"REMOVED PEOPLES = {int(r_t)}\n")


    s_tracker.append(int(s_t))
    i_tracker.append(int(i_t))
    r_tracker.append(int(r_t))


i_tracker_array = np.array(i_tracker)
print(f"The disease lasts {days} days and reachs the max infected peoples: {max(i_tracker)} at days {np.argmax(i_tracker_array)}")


# save all in a file
datafile = open('Lab4/SIRmodel_numerical.dat', 'w')
print('day\tS_t\tI_t\tR_t', file=datafile)

for i in range(len(s_tracker)):
    print(
        i, # the day
        s_tracker[i], # s_t
        i_tracker[i], # i_t
        r_tracker[i], # r_t
        sep='\t',
        file=datafile
    )