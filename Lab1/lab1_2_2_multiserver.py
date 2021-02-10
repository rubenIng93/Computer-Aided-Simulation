import random
from queue import PriorityQueue
from scipy.stats import t
import numpy as np
import pandas as pd
import os

# useful variables
runs = 10
sim_time = 1000
mu_service = {'s1': 3, 's2': 4} # multiserver with different service time
uni_param = {'s1':[1,3], 's2': [3,5]} # parameter for the uniform distribution of the service time
# for this simulation 'a' kept to 1 and varied only b for the load
lambda_arrival = 5 # arrival rate, parameter of Poisson distr.
confidence_level = 0.95
waiting_line = 2 # specify a number for finite capacity
# set the waiting line as empty string for infinite capacity queue
n_servers = 2 # assumed to be 2 for this problem
uniform = True
policy = 'faster'

if uniform == False:
    load = (1/lambda_arrival)/(1/mu_service['s1']+1/mu_service['s2']) # min of 2 exp
    
# DATA STRUCTURES FOR MULTIPLE RUNS
running_data = [] # keep track of the n runs statistics
running_users = [] # keep track of the users during each run
running_len_queue = [] # keep track the actual sizes
running_calls = [] # keep track of the calls for each server

# PRINT INITIAL SETTINGS
print('*'*10 +' INITIAL SETTINGS '+10*'*')
if uniform:
    print(f'Service times uniformly distributed ({uni_param})')
else:
    print(f'Service time: {mu_service}')
    print(f'Load: {load * 100} %')    
print(f'Number of servers: {n_servers}')
print(f'Simulation time: {sim_time}')
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

# class which tracks the servers activities
class Server:
    def __init__(self):
        self.busy = False
        self.calls = 0
        self.departure_time = 0

    def set_busy(self):
        self.busy = True
    
    def set_idle(self):
        self.busy = False
    
    def get_busy(self):
        return self.busy

    def add_call(self):
        self.calls += 1
    
    def get_calls(self):
        return self.calls

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
    if waiting_line != '' and users == waiting_line + 1: # considering also the user in service
        data.loss += 1 # the user leaves the queue without entering
    else:
        users += 1
        # the client here is idetified with the time he arrives
        client = time        
        queue.append(client)

    # if a server is idle start a new service
    if users == 1 : # means that all the servers are idle         
        if uniform:
            # UNIFORM
            # sample the service time uniform or exp
            if policy == 'random':
                rnd = random.randint(1,2) # rnd choice between the server
                chosen_server = f's{rnd}' # s1 or s2
                service_time = random.uniform(uni_param[chosen_server][0] ,uni_param[chosen_server][1])
                # set the chosen server busy and increment its calls
                if rnd == 1:
                    s1.add_call()
                    s1.set_busy()
                    s1.departure_time = time + service_time
                else:
                    s2.add_call()
                    s2.set_busy()
                    s2.departure_time = time + service_time

            elif policy == 'faster':
                # choose always the fastest one
                fastest = np.argmin(tuples_mean(uni_param['s1'], uni_param['s2'])) # track the fastest is 0 or 1
                str_fast = f's{fastest + 1}' # make it as a string
                service_time = random.uniform(uni_param[str_fast][0], uni_param[str_fast][1])

                # make the server busy
                if fastest + 1 == 1:
                    s1.add_call()
                    s1.set_busy()
                    s1.departure_time = time + service_time
                else:
                    s2.add_call()
                    s2.set_busy()
                    s2.departure_time = time + service_time
        else:
            # EXPONENTIAL
            if policy == 'random':
                rnd = random.randint(1,2) # rnd choice between the server
                chosen_server = f's{rnd}' # s1 or s2
                service_time = random.expovariate(1/mu_service[chosen_server])
                # set the chosen server busy and increment its calls
                if rnd == 1:
                    s1.add_call()
                    s1.set_busy()
                    s1.departure_time = time + service_time
                else:
                    s2.add_call()
                    s2.set_busy()
                    s2.departure_time = time + service_time

            elif policy == 'faster':
                # choose always the fastest one
                fastest = np.argmin(list(mu_service.values())) # track the fastest is 0 or 1
                str_fast = f's{fastest + 1}' # make it as a string
                service_time = random.expovariate(1/mu_service[str_fast])

                # make the server busy
                if fastest + 1 == 1:
                    s1.add_call()
                    s1.set_busy()
                    s1.departure_time = time + service_time
                else:
                    s2.add_call()
                    s2.set_busy()
                    s2.departure_time = time + service_time

        # schedule the departure        
        FES.put((time + service_time, 'departure'))
                
    elif users == 2:
        # there is a client in service and one in the queue -> a server idle
        # the client in this case is sent to automatically to the idle one
        if uniform:
            if s1.get_busy():
                # the new client goes to s2
                service_time = random.uniform(uni_param['s2'][0], uni_param['s2'][1])
                s2.set_busy()
                s2.add_call()
                s2.departure_time = time + service_time
            
            else:
                service_time = random.uniform(uni_param['s1'][0], uni_param['s1'][1])
                s1.set_busy()
                s1.add_call()
                s1.departure_time = time + service_time

        else:
            # EXPONENTIAL
            if s1.get_busy():
                # the new client goes to s2
                service_time = random.expovariate(1/mu_service['s2'])
                s2.set_busy()
                s2.add_call()
                s2.departure_time = time + service_time
                
            else:
                service_time = random.expovariate(1/mu_service['s1'])
                s1.set_busy()
                s1.add_call()
                s1.departure_time = time + service_time
            
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
    # set the server which finished its work to idle
    # server tracked wrt time
    if s1.departure_time == time:
        # make it idle
        s1.set_idle()
    else:
        # is the 2nd that will become idle
        s2.set_idle()

    # check whether there are more clients to in the queue
    # considering that a user is already in other servers in service
    if users >= n_servers:
        # sample the service time exp or uniform
        if uniform:
            if s1.get_busy():
                # the new client goes to s2
                service_time = random.uniform(uni_param['s2'][0], uni_param['s2'][1])
                s2.set_busy()
                s2.add_call()
            
            else:
                service_time = random.uniform(uni_param['s1'][0], uni_param['s1'][1])
                s1.set_busy()
                s1.add_call()
        else:
            # EXPONENTIAL
            if s1.get_busy():
                # the new client goes to s2
                service_time = random.expovariate(1/mu_service['s2'])
                s2.set_busy()
                s2.add_call()
            
            else:
                service_time = random.expovariate(1/mu_service['s1'])
                s1.set_busy()
                s1.add_call()
                
                # schedule the departure
        FES.put((time + service_time, 'departure'))             

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

# function to find the fastest uniform distribution
def tuples_mean(tuple_a, tuple_b):
    mean1 = (tuple_a[0] + tuple_a[1]) / 2
    mean2 = (tuple_b[0] + tuple_b[1]) / 2

    return mean1, mean2


# SETTING THE SEED
random.seed(22)

# RUNS LOOP
for _ in range(runs): 
    # INITIALIZATION
    users = 0
    time = 0
    queue = []
    s1 = Server() # call the class server
    s2 = Server() # call the class server
    data = Measures(0,0,0,0,0,0)
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
    running_calls.append((s1.calls, s2.calls))


# EVALUATE AND THE CIs FOR THE STATISTICS
# make sense only for the avg users and avg delay
delays = np.zeros(runs) # initialize np array
avg_users = np.zeros(runs) # initialize np array
arrivals = np.zeros(runs) # initialize np array
departures = np.zeros(runs) # initialize np array
losses = np.zeros(runs) # initialize np array
s1_calls = np.zeros(runs) # initialize np array
s2_calls = np.zeros(runs) # initialize np array

# conversion in numpy array
for i in range(runs):
    data = running_data[i]
    arrivals[i] = data.arr
    departures[i] = data.dep
    avg_users[i] = data.ut
    delays[i] = data.delay/data.dep
    losses[i] = data.loss
    s1_calls[i] = running_calls[i][0]
    s2_calls[i] = running_calls[i][1]

avg_delays, ci_delays, err_delays = evaluate_conficence_interval(delays) # delays
avg_us, ci_users, err_users = evaluate_conficence_interval(avg_users) # avg users
avg_tot_users, ci_tot_users, err_tot_users = evaluate_conficence_interval(np.array(running_users)) # tot users
avg_dep, ci_dep, err_dep = evaluate_conficence_interval(departures) # departures
avg_arr, ci_arr, err_arr = evaluate_conficence_interval(arrivals) # arrivals
avg_len_queue, ci_len_queue, err_len_queue = evaluate_conficence_interval(np.array(running_len_queue)) # residual queue
avg_loss, ci_loss, err_loss = evaluate_conficence_interval(losses) # losses
avg_s1_calls, ci_s1_calls, err_s1_calls = evaluate_conficence_interval(s1_calls) # s1 calls
avg_s2_calls, ci_s2_calls, err_s2_calls = evaluate_conficence_interval(s2_calls) # s2 calls

# PRINT OUTPUT ORGANIZED IN PANDAS DATAFRAME

df = pd.DataFrame(data={
    'MEASURES':['Res users', 'Avg users', 'Arrivals', 'Departures', 'Delay','Res queue', 'Losses', 'S1 Calls', 'S2 Calls'],
    'Lower bound':[avg_tot_users-ci_tot_users, (avg_us-ci_users)/sim_time, avg_arr-ci_arr, avg_dep-ci_dep, avg_delays-ci_delays, \
        avg_len_queue-ci_len_queue, avg_loss-ci_loss, avg_s1_calls-ci_s1_calls, avg_s2_calls-ci_s2_calls],
    'Mean':[avg_tot_users, avg_us/sim_time, avg_arr, avg_dep, avg_delays, avg_len_queue, avg_loss, avg_s1_calls, avg_s2_calls],
    'Upper bound':[avg_tot_users+ci_tot_users, (avg_us+ci_users)/sim_time, avg_arr+ci_arr, avg_dep+ci_dep, avg_delays+ci_delays, \
        avg_len_queue+ci_len_queue, avg_loss+ci_loss, avg_s1_calls+ci_s1_calls, avg_s2_calls+ci_s2_calls],
    'Relative error':[err_tot_users, err_users, err_arr, err_dep, err_delays, err_len_queue, err_loss, err_s1_calls, err_s2_calls]
})

print("\n","*"*10,"  MEASUREMENTS  ","*"*10,"\n")
print(df.set_index('MEASURES'))
print("\n","*"*40)

# PRINT COMPARISON THEORICAL/EMPIRICAL
print("\n","*"*10,"  COMPARISON THEORICAL/EMPIRICAL  ","*"*10,"\n")
print("Nominal arrival rate: ",1.0/lambda_arrival)
print("Avg measured arrival rate",avg_arr/time,"\nAvg measured departure rate: ",avg_dep/time)
print("\n","*"*40)

# SAVE IT IN A FILE
options = {
    'sep': '\t',
    'index': 'MEASURES'    
}
# create a folder if not present
if os.listdir(os.getcwd()+"/Lab1").__contains__("data") == False:
    os.mkdir(os.getcwd()+"/Lab1/data")

if uniform:
    if waiting_line == '': 
        df.to_csv(f'Lab1/data/uni_service_MM2_{policy}.csv', **options)
    else:
        df.to_csv(f'Lab1/data/uni_service_MM2B_{policy}.csv', **options)
else:
    if waiting_line == '':  
        df.to_csv(f'Lab1/data/exp_service_MM2_{policy}.csv', **options)
    else:
        df.to_csv(f'Lab1/data/exp_service_MM2B_{policy}.csv', **options)


