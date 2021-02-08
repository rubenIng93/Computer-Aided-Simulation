import random
from queue import PriorityQueue
from scipy.stats import t
import numpy as np
import pandas as pd

# useful variables
runs = 10
sim_time = 1000
mu_service = 4.5 # service rate, parameter of the exp distribution
uni_param = {'a':1, 'b': 8.8} # parameter for the uniform distribution of the service time
# for this simulation 'a' kept to 1 and varied only b for the load
lambda_arrival = 5 # arrival rate, parameter of Poisson distr.
confidence_level = 0.95
uniform = True

if uniform:
    mean = (uni_param['a'] + uni_param['b']) / 2
    load = mean / lambda_arrival
    mu_service = mean
else:
    load = mu_service/lambda_arrival

# DATA STRUCTURES FOR MULTIPLE RUNS
running_data = [] # keep track of the n runs statistics
running_users = [] # keep track of the users during each run
running_len_queue = [] # keep track the actual sizes
# print the initial settings

print('*'*10 +' INITIAL SETTINGS '+10*'*')
if uniform:
    print(f'Service times uniformly distributed ({uni_param})')
else:
    print(f'Service times exponentially distributed ({1/mu_service})')
    
print(f'Load: {load * 100} %')
print(f'Simulation time: {sim_time}')
print(f'# Runs: {runs}\nArrival rate: {1/lambda_arrival}')
print(f'Confidence level: {confidence_level*100} %\n')
print(40 * '*')

# define a class which tracks the measures
class Measures:
    def __init__(self,Narr,Ndep,NAveraegUser,OldTimeEvent,AverageDelay):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay

# function that triggers when the arrival event happens
def arrival(time, FES, queue):
    global users

    # cumalate statistics
    data.arr += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time

    # sample the interarrival time
    inter_arrival = random.expovariate(1/lambda_arrival)

    # schedule the next arrival
    FES.put((time + inter_arrival, 'arrival'))

    # update the state variable
    users += 1

    # the client here is idetified with the time he arrives
    client = time
    
    queue.append(client)

    # if the server is idle start a new service
    if users == 1:
        # sample the service time uniform or exp
        if uniform:
            service_time = random.uniform(uni_param['a'], uni_param['b'])
        else:
            service_time = random.expovariate(1/mu_service)
        # schedule the departure
        FES.put((time + service_time, 'departure'))

# function that triggers when the departure event happens
def departure(time, FES, queue):
    global users

    # get the first element from the queue
    client = queue.pop(0)

    # cumulate statistics
    data.dep += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time
    data.delay += (time-client)

    # update the state variable, by decreasing the no. of clients by 1
    users -= 1

    # check whether there are more clients to in the queue
    if users > 0:
        # sample the service time exp or uniform
        if uniform:
            service_time = random.uniform(uni_param['a'], uni_param['b'])
        else:
            service_time = random.expovariate(1/mu_service)

        # schedule the departure of the client
        FES.put((time + service_time, "departure"))


# function that computes the confidence interval
def evaluate_conficence_interval(values):
    t_treshold = t.ppf((confidence_level + 1) / 2, df= runs-1)
    ave = values.mean()
    stddev = values.std(ddof=1)
    ci = t_treshold * stddev /np.sqrt(runs)
    rel_err = ci/ ave
    return ave, ci, rel_err

# function that derives E[N] and E[T] for the uniform distribution selected
def pollaczek_khintchin(uni_dict):

    # compute the coefficient of variation
    var = (uni_dict['b'] - uni_dict['a'])**2 / 12
    c = var / mean**2

    E_N = (1/lambda_arrival)*mean + ((1/lambda_arrival)*mean)**2 * (1+c)/(2*(1-mean/lambda_arrival))
    E_T = E_N / (1/lambda_arrival)

    return E_N, E_T


# SETTING THE SEED
random.seed(22)

# RUNS LOOP
for _ in range(runs): 
    # INITIALIZATION
    users = 0
    time = 0
    queue = []
    data = Measures(0,0,0,0,0)
    FES = PriorityQueue()
    FES.put((0, 'arrival')) # schedule the first event

    # EVENT LOOP
    while time < sim_time:
        # extract next event from the FES
        time, event_type = FES.get()
        # call the right event function
        if event_type == 'arrival':
            arrival(time, FES, queue)

        elif event_type == 'departure':
            departure(time, FES, queue)

    # save the statistics
    running_data.append(data)
    running_users.append(users)
    running_len_queue.append(len(queue))



# EVALUATE AND THE CIs FOR THE STATISTICS
# make sense only for the avg users and avg delay
delays = np.zeros(runs) # initialize np array
avg_users = np.zeros(runs) # initialize np array
arrivals = np.zeros(runs) # initialize np array
departures = np.zeros(runs) # initialize np array

for i in range(runs):
    data = running_data[i]
    arrivals[i] = data.arr
    departures[i] = data.dep
    avg_users[i] = data.ut
    delays[i] = data.delay/data.dep

avg_delays, ci_delays, err_delays = evaluate_conficence_interval(delays) # delays
avg_us, ci_users, err_users = evaluate_conficence_interval(avg_users) # avg users
avg_tot_users, ci_tot_users, err_tot_users = evaluate_conficence_interval(np.array(running_users)) # tot users
avg_dep, ci_dep, err_dep = evaluate_conficence_interval(departures) # departures
avg_arr, ci_arr, err_arr = evaluate_conficence_interval(arrivals) # arrivals
avg_len_queue, ci_len_queue, err_len_queue = evaluate_conficence_interval(np.array(running_len_queue)) # residual queue

# PRINT OUTPUT ORGANIZED IN PANDAS DATAFRAME

df = pd.DataFrame(data={
    'MEASURES':['Res users', 'Avg users', 'Arrivals', 'Departures', 'Delay','Res queue'],
    'Lower bound':[avg_tot_users-ci_tot_users, (avg_us-ci_users)/sim_time, avg_arr-ci_arr, avg_dep-ci_dep, avg_delays-ci_delays, avg_len_queue-ci_len_queue],
    'Mean':[avg_tot_users, avg_us/sim_time, avg_arr, avg_dep, avg_delays, avg_len_queue],
    'Upper bound':[avg_tot_users+ci_tot_users, (avg_us+ci_users)/sim_time, avg_arr+ci_arr, avg_dep+ci_dep, avg_delays+ci_delays, avg_len_queue+ci_len_queue],
    'Relative error':[err_tot_users, err_users, err_arr, err_dep, err_delays, err_len_queue]
})

print("\n","*"*10,"  MEASUREMENTS  ","*"*10,"\n")
print(df.set_index('MEASURES'))
print("\n","*"*40)

# SAVE IT IN A FILE
options = {
    'sep': '\t',
    'index': 'MEASURES'    
}
if uniform: 
    df.to_csv(f'Lab1/uni_service_{int(load*100)}load.csv', **options)
else: 
    df.to_csv(f'Lab1/exp_service_{int(load*100)}load.csv', **options)


# PRINT COMPARISON THEORICAL/EMPIRICAL
print("\n","*"*10,"  COMPARISON THEORICAL/EMPIRICAL  ","*"*10,"\n")
print("Nominal arrival rate: ",1.0/lambda_arrival)
print("Avg measured arrival rate",avg_arr/time,"\nAvg measured departure rate: ",avg_dep/time)
if uniform:
    theorical_n, theorical_t = pollaczek_khintchin(uni_param) # E[N], E[T] for MG1 queues
else:
    theorical_n=(1.0/lambda_arrival)/(1.0/mu_service-1.0/lambda_arrival) # E[N] for MM1 queue
    theorical_t=1.0/(1.0/mu_service-1.0/lambda_arrival) # E[T] from MM1 queue

print("\n\nAverage number of users\nTheorical: ", theorical_n,\
      "  -  Avg empirical: ",avg_us/time)
print("Average delay \nTheorical= ",theorical_t,"  -  Avg empirical: ",avg_delays)

print("\n","*"*40)
