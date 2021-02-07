import random
import numpy as np

# queuing system with 1 server, inf capacity, FIFO discipline
# arrival Poisson

# usefull variables
runs = 10
total_time = 1000 # duration
lambda_exp = 0.5 # [1/s] parameter for exponential distribution, service rate = mu
lambda_poisson = 0.2 # arrival rate = lambda
a_uni = 0.1
b_uni = 2.1

# setting the seeds
np.random.seed(22)

# 1) service time exponentially distributed

clients = [] # array of clients that are waiting
server_idle = True # whether on not the server is busy
time = 0
service_times = []
service_times_uniform = []
clients_served = 0

while time < total_time:
    arrival = np.random.poisson(lambda_poisson)
    if time + arrival > total_time:
        break
    else:
        time += arrival
        clients.append(time)

print(f'The queue looks {len(clients)} clients in {total_time} seconds')

time = clients[0]
while time < total_time:
    if server_idle:
        # schedule a service time
        server_idle = False
        service_time = np.random.exponential(1/lambda_exp)
        service_uniform = np.random.uniform(a_uni, b_uni)
        #print(service_time)
        if time + service_time > total_time:
            break
        service_times.append(service_time + time)
        service_times_uniform.append(service_uniform)
        clients_served += 1
        time += 0.001

    else:
        time += 0.001
        # check if in the time + 1 the server is idle
        if service_times[-1] <= time:
            server_idle = True

print('Clients served = ', clients_served)


print('ANALYTICAL PERFORMANCES')

ro_exp = lambda_poisson / lambda_exp
print(f'λ = {lambda_poisson}, μ = {lambda_exp}')
print('ρ = ', ro_exp)

E_N = ro_exp / (1 - ro_exp)
E_Nw = ro_exp ** 2 / (1 - ro_exp)
E_T = 1 / (lambda_exp - lambda_poisson)
E_Tw = ro_exp / (lambda_exp - lambda_poisson)

print('Avg customers in the queue = ', E_N)
print('Avg customers in the waiting line = ', E_Nw)
print(f'Avg time in the queue = {E_T} s')
print(f'Avg time in the waiting line = {E_Tw} s')