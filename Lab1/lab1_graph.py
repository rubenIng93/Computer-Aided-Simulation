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
                if row[10] != 'None':
                    theo.append(float(row[10]))
                else:
                    theo.append(0)

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
                if row[11] != 'None':
                    theo.append(float(row[11]))
                else:
                    theo.append(0)

    return loads, lb, mean, ub, theo

def pick_abs_loss_values(in_file):
    loads = []
    lb = []
    mean = []
    ub = []

    with open(in_file, 'r') as _file:
        plots = csv.reader(_file, delimiter='\t')
        header = True
        for row in plots:
            if header:
                header = False
            else:
                loads.append(float(row[0])*100)
                lb.append(float(row[7]))
                mean.append(float(row[8]))
                ub.append(float(row[9]))                

    return loads, lb, mean, ub

def pick_prob_loss_values(in_file):
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
                lb.append(float(row[7])*100)
                mean.append(float(row[8])*100)
                ub.append(float(row[9])*100)   
                theo.append(float(row[12])*100)             

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

    elif _type == 'loss':
        pass

    elif _type == 'loss_probs':

        B = list(input_filename)[13] # retrieve the capacity of waiting line
        # considering also the path

        loads1, lb1, mean1, ub1, theo1 = pick_prob_loss_values(input_filename)

        plt.plot(loads1, mean1, label=f'MM1{B} simulation')
        plt.plot(loads1, theo1, label=f'MM1{B} theorical value', linestyle='dotted')
        plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label=f'MM1{B} 95% CI')
        plt.xlabel('Loads [%]')
        plt.ylabel('Loss probability [%]')
        
        title = f'Loss probability in different conditions of service time MM1{B} queue'
        plt.title('mean of inter-arrival time fixed at 5s')

        if comparison != '':
            loads2, lb2, mean2, ub2, theo2 = pick_prob_loss_values(comparison)
            plt.plot(loads2, mean2, label=f'MG1{B} simulation')
            plt.fill_between(loads2, lb2, ub2, color='g', alpha=.1, label=f'MG1{B} 95% CI')
            title = f'Loss probability w.r.t. loads. Comparison MM1{B}/MG1{B}'

        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()
        
def mx1b_metrics(input_filename, out_filename, _type, comparison=''):

    B = list(input_filename)[13] # retrieve the capacity of waiting line
        # considering also the path

    if _type == 'delay':
        loads1, lb1, mean1, ub1, theo1 = pick_delay_values(input_filename)

        plt.plot(loads1, mean1, label=f'MM1{B} simulation')
        plt.plot(loads1, theo1, label=f'MM1{B} theorical value', linestyle='dotted')
        plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label=f'MM1{B} 95% CI')
        plt.xlabel('Loads [%]')
        plt.ylabel('Delay [s]')
        title = 'E[T] in different conditions of service time MM1B queue'
        plt.title('mean of inter-arrival time fixed at 5s')

        if comparison != '':
            loads2, lb2, mean2, ub2, _ = pick_delay_values(comparison)
            plt.plot(loads2, mean2, label=f'MG1{B} simulation')
            #plt.plot(loads2, theo2, label='MG1 theorical value', linestyle='dotted')
            plt.fill_between(loads2, lb2, ub2, color='g', alpha=.1, label=f'MG1{B} 95% CI')
            title = f'E[T] w.r.t. loads. Comparison MM1{B}/MG1{B}'

        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()

    elif _type == 'avg_users':
        loads1, lb1, mean1, ub1, theo1 = pick_avg_cust_values(input_filename)

        plt.plot(loads1, mean1, label=f'MM1{B} simulation')
        plt.plot(loads1, theo1, label=f'MM1{B} theorical value', linestyle='dotted')
        plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label=f'MM1{B} 95% CI')
        plt.xlabel('Loads [%]')
        plt.ylabel('Avg users in the queue')
        title = f'E[N] in different conditions of service time MM1{B} queue'
        plt.title('mean of inter-arrival time fixed at 5s')

        if comparison != '':
            loads2, lb2, mean2, ub2, _ = pick_avg_cust_values(comparison)
            plt.plot(loads2, mean2, label=f'MG1{B} simulation')
            #plt.plot(loads2, theo2, label=f'MG1{B} theorical value', linestyle='dotted')
            plt.fill_between(loads2, lb2, ub2, color='g', alpha=.1, label=f'MG1{B} 95% CI')
            title = f'E[N] w.r.t. loads. Comparison MM1{B}/MG1{B}'

        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()

#metric_wrt_mu_service('Lab1/data/prova.dat', 'exp_delay_wrt_mu_MM1.png', 'delay')
metric_wrt_mu_service('Lab1/data/MM1.dat', 'comparison_delay_Mx1.png', 'delay', comparison='Lab1/data/MG1.dat')
metric_wrt_mu_service('Lab1/data/MM1.dat', 'comparison_ET_Mx1.png', 'avg_users', comparison='Lab1/data/MG1.dat')
#mx12
metric_wrt_mu_service('Lab1/data/MM12.dat', 'comparison_loss_Mx12.png', 'loss_probs', comparison='Lab1/data/MG12.dat')
mx1b_metrics('Lab1/data/MM12.dat', 'comparison_delay_Mx12.png', 'delay', comparison='Lab1/data/MG12.dat')
mx1b_metrics('Lab1/data/MM12.dat', 'comparison_users_Mx12.png', 'avg_users', comparison='Lab1/data/MG12.dat')
# mx15
metric_wrt_mu_service('Lab1/data/MM15.dat', 'comparison_loss_Mx15.png', 'loss_probs', comparison='Lab1/data/MG15.dat')
mx1b_metrics('Lab1/data/MM15.dat', 'comparison_delay_Mx15.png', 'delay', comparison='Lab1/data/MG15.dat')
mx1b_metrics('Lab1/data/MM15.dat', 'comparison_users_Mx15.png', 'avg_users', comparison='Lab1/data/MG15.dat')
#mx110
metric_wrt_mu_service('Lab1/data/MM110.dat', 'comparison_loss_Mx110.png', 'loss_probs', comparison='Lab1/data/MG110.dat')
mx1b_metrics('Lab1/data/MM110.dat', 'comparison_delay_Mx110.png', 'delay', comparison='Lab1/data/MG110.dat')
mx1b_metrics('Lab1/data/MM110.dat', 'comparison_users_Mx110.png', 'avg_users', comparison='Lab1/data/MG110.dat')


    


