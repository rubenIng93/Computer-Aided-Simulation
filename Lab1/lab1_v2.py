import random
from queue import PriorityQueue
from scipy.stats import t
import numpy as np
import os

# useful variables
runs = 10
sim_time = 10000
mu_service = 4.5 # service rate, parameter of the exp distribution
uni_param = {'a':1, 'b': 3} # parameter for the uniform distribution of the service time
# for this simulation 'a' kept to 1 and varied only b for the load
lambda_arrival = 5 # arrival rate, parameter of Poisson distr.
confidence_level = 0.95
waiting_line = 20 # specify a number for finite capacity or a list for check the impact 
# with the function on purpose
# set the waiting line as empty string for infinite capacity queue
n_servers = 1 # set to integer > 1 to exploit the multi server approach
uniform = True

# print the initial settings

print('*'*10 +' INITIAL SETTINGS '+10*'*')
if uniform:
    print(f'Simulation with service times uniformly distributed')
else:
    print(f'Simulation with service times exponentially distributed)')
    
print(f'Number of servers: {n_servers}')
print(f'Simulation time: {sim_time} s')
print(f'# Runs: {runs}\nArrival rate: {1/lambda_arrival}')
print(f'Confidence level: {confidence_level*100} %\n')
print(40 * '*')

# define a class which tracks the measures
class Measures:
    def __init__(self,Narr,Ndep,NAveraegUser,OldTimeEvent,AverageDelay, Losses):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay
        self.loss = Losses # in case of finite capacity of the waiting line

# function that triggers when the arrival event happens
def arrival(time, FES, queue, param):
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
    if waiting_line != '' and users == waiting_line + 1: # considering also the user in service
        data.loss += 1 # the user leaves the queue without entering
    else:
        users += 1
        # the client here is idetified with the time he arrives
        client = time        
        queue.append(client)

    # if the server is idle start a new service
    if users <= n_servers and len(queue) > 0:
        # sample the service time uniform or exp
        if uniform:
            service_time = random.uniform(uni_param['a'], param)
        else:
            service_time = random.expovariate(1/param)
        # schedule the departure
        FES.put((time + service_time, 'departure'))

# function that triggers when the departure event happens
def departure(time, FES, queue, param):
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
    # considering that a user is already in other servers in service
    if users >= n_servers:
        # sample the service time exp or uniform
        if uniform:
            service_time = random.uniform(uni_param['a'], param)
        else:
            service_time = random.expovariate(1/param)

        # schedule the departure of the client
        FES.put((time + service_time, "departure"))

# function that computes the confidence interval
def evaluate_conficence_interval(values):
    t_treshold = t.ppf((confidence_level + 1) / 2, df= runs-1)
    ave = values.mean()
    stddev = values.std(ddof=1)
    ci = t_treshold * stddev /np.sqrt(runs)
    if ave != 0:
        rel_err = ci/ ave
    else:
        rel_err = None
    return ave, ci, rel_err

# function that derives E[N] and E[T] for the uniform distribution selected
def pollaczek_khintchin(uni_dict, param):

    # compute the coefficient of variation
    var = (param - uni_dict['a'])**2 / 12
    c = var / mean**2
    print(f'Coefficient of variation: C = {c:.2f}')

    E_N = (1/lambda_arrival)*mean + ((1/lambda_arrival)*mean)**2 * (1+c)/(2*(1-mean/lambda_arrival))
    E_T = E_N / (1/lambda_arrival)

    return E_N, E_T


# SETTING THE SEED
random.seed(22)
loads_delay = [] # data str that tracks the delay for different loads
loads_avg_users = [] # same for avg_users
loads_loss = [] # same for losses
loads_theory = [] # same for theorical formulas
loads_loss_theo = [] # same for loss probability theo
debug = 0 # each 10 iteration an output

if uniform:
    params = np.linspace(1.01, 8.9, 50) # b param of the uniform that changes
    # according to different loads
    loads = []  # list that tracks the loas for each b
else:
    params = np.linspace(0.2, 4.9, 50) # service times

# LOOP FOR DIFFERENT SERVICE TIME

for param in params:   
   
    # RUNS LOOP
    # DATA STRUCTURES FOR MULTIPLE RUNS
    running_data = [] # keep track of the n runs statistics
    running_users = [] # keep track of the users during each run
    running_len_queue = [] # keep track the actual sizes
    
    for _ in range(runs): 

        # INITIALIZATION
        users = 0
        time = 0
        queue = []
        data = Measures(0,0,0,0,0,0)
        FES = PriorityQueue()
        FES.put((0, 'arrival')) # schedule the first event

        # EVENT LOOP
        while time < sim_time:
            # extract next event from the FES
            time, event_type = FES.get()
            # call the right event function
            if event_type == 'arrival':
                arrival(time, FES, queue, param)

            elif event_type == 'departure':
                departure(time, FES, queue, param)

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
    losses = np.zeros(runs) # initialize np array

    for i in range(runs):
        data = running_data[i]
        arrivals[i] = data.arr
        departures[i] = data.dep
        avg_users[i] = data.ut/sim_time
        delays[i] = data.delay/data.dep
        losses[i] = data.loss

    avg_delays, ci_delays, err_delays = evaluate_conficence_interval(delays) # delays
    avg_us, ci_users, err_users = evaluate_conficence_interval(avg_users) # avg users
    avg_tot_users, ci_tot_users, err_tot_users = evaluate_conficence_interval(np.array(running_users)) # tot users
    avg_dep, ci_dep, err_dep = evaluate_conficence_interval(departures) # departures
    avg_arr, ci_arr, err_arr = evaluate_conficence_interval(arrivals) # arrivals
    avg_len_queue, ci_len_queue, err_len_queue = evaluate_conficence_interval(np.array(running_len_queue)) # residual queue
    avg_loss, ci_loss, err_loss = evaluate_conficence_interval(losses) # losses

    # Theorical computation
    if uniform:
        mean = (uni_param['a'] + param) / 2
        load = mean / lambda_arrival /n_servers
        loads.append(load)
        if waiting_line == '' and n_servers == 1: # otherwise not formulas available
            theorical_n, theorical_t = pollaczek_khintchin(uni_param, param) # E[N], E[T] for MG1 queues
        else:
            # no formulas for finite capacity uniform and multiple server
            theorical_n, theorical_t = None, None
        loss_theorical = 0
    else:
        # Markovian
        if waiting_line == '' and n_servers == 1:
            theorical_n=(1.0/lambda_arrival)/(1.0/param-1.0/lambda_arrival) # E[N] for MM1 queue
            theorical_t=1.0/(1.0/param-1.0/lambda_arrival) # E[T] from MM1 queue
            loss_theorical = 0

        elif n_servers > 1:
            theorical_n, theorical_t = None, None
            loss_theorical = 0

        else: # computation for MM1B - only for Markovian
            ro = (1/lambda_arrival)/(1/param*n_servers)
            loss_theorical = (1 - ro) / (1 - ro**(waiting_line+2)) * ro**(waiting_line+1)            

            # computation E[N] theorical
            theorical_n = 0
            for i in range(0, waiting_line+2): # in [1,5] in this case
                theorical_n += (1-ro) / (1-ro**(waiting_line+2)) * ro**i * i 
            e_lambda = (1/lambda_arrival) - (1/lambda_arrival)*loss_theorical
            theorical_t = theorical_n / e_lambda

        load = param/lambda_arrival/n_servers # compute the load


    loads_delay.append((avg_delays-ci_delays, avg_delays, avg_delays+ci_delays)) # append a tuple for the ci
    loads_avg_users.append((avg_us-ci_users, avg_us, avg_us+ci_users))
    loads_loss.append(((avg_loss-ci_loss)/avg_arr, avg_loss/avg_arr, (avg_loss+ci_loss)/avg_arr))
    loads_loss_theo.append(loss_theorical)
    loads_theory.append((theorical_t, theorical_n))
    
    if debug % 10 == 0:
        print(f'Running for LOAD = {load*100:.2f} %')
        print('\n***CURRENT STATS***')
        print("Avg measured arrival rate",avg_arr/time,"\nAvg measured departure rate: ",avg_dep/time)
        print("\n\nAverage number of users\nTheorical: ", theorical_n,\
            "  -  Avg empirical: ",avg_us)
        print("Average delay \nTheorical= ",theorical_t,"  -  Avg empirical: ",avg_delays)
        if uniform:
            print(f'\nLoss probability \nEmpirical= {avg_loss/avg_arr*100} %')
        else:
            print(f'\nLoss probability \nTheorical= {loss_theorical*100} % - Empirical= {avg_loss/avg_arr*100} %')
        print("\n","*"*40)

    debug += 1

# create a folder if not present
if os.listdir(os.getcwd()+"/Lab1").__contains__("data") == False:
    os.mkdir(os.getcwd()+"/Lab1/data")


# SAVE IN A FILE
if uniform:
    if waiting_line == '':
        datafile = open(f'Lab1/data/MG{n_servers}.dat', 'w')
    else:
        datafile = open(f'Lab1/data/MG{n_servers}{waiting_line}.dat', 'w')
    print('Load\tLBDelay\tAvgDelay\tUBDelay\tLBUsers\tAvgUsers\tUBUsers\tLBLosses\tAvgLosses\tUBLosses\tTheoDelay\tTheoUsers\tTheoLoss', file=datafile) # print the header

    for i in range(len(params)):
        print(loads[i], *loads_delay[i], *loads_avg_users[i], *loads_loss[i], *loads_theory[i], loads_loss_theo[i], sep='\t', file=datafile)

    datafile.close()

else:
    if waiting_line == '':
        datafile = open(f'Lab1/data/MM{n_servers}.dat', 'w')
    else:
        datafile = open(f'Lab1/data/MM{n_servers}{waiting_line}.dat', 'w')
    print('Load\tLBDelay\tAvgDelay\tUBDelay\tLBUsers\tAvgUsers\tUBUsers\tLBLosses\tAvgLosses\tUBLosses\tTheoDelay\tTheoUsers\tTheoLoss', file=datafile) # print the header

    for i in range(len(params)):
        print(params[i]/lambda_arrival, *loads_delay[i], *loads_avg_users[i], *loads_loss[i], *loads_theory[i], loads_loss_theo[i], sep='\t', file=datafile)

    datafile.close()
