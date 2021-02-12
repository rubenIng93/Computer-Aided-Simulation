import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv


# make a folder
if os.listdir(os.getcwd()+"/Lab1").__contains__("images") == False:
    os.mkdir(os.getcwd()+"/Lab1/images")


def pick_delay_values(in_file):
    loads = []
    lb = []
    mean = []
    ub = []
    theo = []

    with open(in_file, 'r') as _file:
        plots = csv.reader(_file, delimiter='\t')
        header = True
        for row in plots:
            if header:
                header = False
            else:
                loads.append(float(row[0])*100)
                lb.append(float(row[1]))
                mean.append(float(row[2]))
                ub.append(float(row[3]))
                theo.append(float(row[10]))

    return loads, lb, mean, ub, theo

def pick_avg_cust_values(in_file):
    loads = []
    lb = []
    mean = []
    ub = []
    theo = []

    with open(in_file, 'r') as _file:
        plots = csv.reader(_file, delimiter='\t')
        header = True
        for row in plots:
            if header:
                header = False
            else:
                loads.append(float(row[0])*100)
                lb.append(float(row[4]))
                mean.append(float(row[5]))
                ub.append(float(row[6]))
                theo.append(float(row[11]))

    return loads, lb, mean, ub, theo


def metric_wrt_mu_service(input_filename, out_filename, _type, comparison=''):

    if _type == 'delay':
        loads1, lb1, mean1, ub1, theo1 = pick_delay_values(input_filename)

        plt.plot(loads1, mean1, label='MM1 simulation')
        plt.plot(loads1, theo1, label='MM1 theorical value', linestyle='dotted')
        plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label='MM1 95% CI')
        plt.xlabel('Loads [%]')
        plt.ylabel('Delay [s]')
        title = 'E[T] in different conditions of service time MM1 queue'
        plt.title('mean of inter-arrival time fixed at 5s')

        if comparison != '':
            loads2, lb2, mean2, ub2, theo2 = pick_delay_values(comparison)
            plt.plot(loads2, mean2, label='MG1 simulation')
            plt.plot(loads2, theo2, label='MG1 theorical value', linestyle='dotted')
            plt.fill_between(loads2, lb2, ub2, color='g', alpha=.1, label='MG1 95% CI')
            title = 'E[T] w.r.t. loads. Comparison MM1/MG1'

        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()

    elif _type == 'avg_users':
        loads1, lb1, mean1, ub1, theo1 = pick_avg_cust_values(input_filename)

        plt.plot(loads1, mean1, label='MM1 simulation')
        plt.plot(loads1, theo1, label='MM1 theorical value', linestyle='dotted')
        plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label='MM1 95% CI')
        plt.xlabel('Loads [%]')
        plt.ylabel('Avg users in the queue')
        title = 'E[N] in different conditions of service time MM1 queue'
        plt.title('mean of inter-arrival time fixed at 5s')

        if comparison != '':
            loads2, lb2, mean2, ub2, theo2 = pick_avg_cust_values(comparison)
            plt.plot(loads2, mean2, label='MG1 simulation')
            plt.plot(loads2, theo2, label='MG1 theorical value', linestyle='dotted')
            plt.fill_between(loads2, lb2, ub2, color='g', alpha=.1, label='MG1 95% CI')
            title = 'E[N] w.r.t. loads. Comparison MM1/MG1'

        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()
    


#metric_wrt_mu_service('Lab1/data/prova.dat', 'exp_delay_wrt_mu_MM1.png', 'delay')
metric_wrt_mu_service('Lab1/data/MM1.dat', 'comparison_delay_Mx1.png', 'delay', comparison='Lab1/data/MG1.dat')
metric_wrt_mu_service('Lab1/data/MM1.dat', 'comparison_ET_Mx1.png', 'avg_users', comparison='Lab1/data/MG1.dat')
    


