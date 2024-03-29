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

    elif _type == 'loss_probs':

        B = list(input_filename)[13] # retrieve the capacity of waiting line
        # considering also the path

        loads1, lb1, mean1, ub1, theo1 = pick_prob_loss_values(input_filename)

        plt.plot(loads1, mean1, label=f'MM2{B} simulation')
        #plt.plot(loads1, theo1, label=f'MM1{B} theorical value', linestyle='dotted')
        plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label=f'MM2{B} 95% CI')
        plt.xlabel('Loads [%]')
        plt.ylabel('Loss probability [%]')
        
        title = f'Loss probability in different conditions of service time MM2{B} queue'
        plt.title('mean of inter-arrival time fixed at 5s')

        if comparison != '':
            loads2, lb2, mean2, ub2, theo2 = pick_prob_loss_values(comparison)
            plt.plot(loads2, mean2, label=f'MG2{B} simulation', color='g')
            plt.fill_between(loads2, lb2, ub2, color='g', alpha=.1, label=f'MG2{B} 95% CI')
            title = f'Loss probability w.r.t. loads. Comparison MM2{B}/MG2{B}'

        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()
        
def mx1b_metrics(input_filename, out_filename, _type, comparison=''):

    B = list(input_filename)[13] # retrieve the capacity of waiting line
        # considering also the path

    if _type == 'delay':
        loads1, lb1, mean1, ub1, _ = pick_delay_values(input_filename)

        plt.plot(loads1, mean1, label=f'MM2{B} simulation')
        #plt.plot(loads1, theo1, label=f'MM1{B} theorical value', linestyle='dotted')
        plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label=f'MM2{B} 95% CI')
        plt.xlabel('Loads [%]')
        plt.ylabel('Delay [s]')
        title = 'E[T] in different conditions of service time MM12 queue'
        plt.title('mean of inter-arrival time fixed at 5s')

        if comparison != '':
            loads2, lb2, mean2, ub2, _ = pick_delay_values(comparison)
            plt.plot(loads2, mean2, label=f'MG2{B} simulation', color='g')
            #plt.plot(loads2, theo2, label='MG1 theorical value', linestyle='dotted')
            plt.fill_between(loads2, lb2, ub2, color='g', alpha=.1, label=f'MG2{B} 95% CI')
            title = f'E[T] w.r.t. loads. Comparison MM2{B}/MG2{B}'

        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()

    elif _type == 'avg_users':
        loads1, lb1, mean1, ub1, theo1 = pick_avg_cust_values(input_filename)

        plt.plot(loads1, mean1, label=f'MM2{B} simulation')
        #plt.plot(loads1, theo1, label=f'MM1{B} theorical value', linestyle='dotted')
        plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label=f'MM2{B} 95% CI')
        plt.xlabel('Loads [%]')
        plt.ylabel('Avg users in the queue')
        title = f'E[N] in different conditions of service time MM2{B} queue'
        plt.title('mean of inter-arrival time fixed at 5s')

        if comparison != '':
            loads2, lb2, mean2, ub2, _ = pick_avg_cust_values(comparison)
            plt.plot(loads2, mean2, label=f'MG2{B} simulation', color='g')
            #plt.plot(loads2, theo2, label=f'MG1{B} theorical value', linestyle='dotted')
            plt.fill_between(loads2, lb2, ub2, color='g', alpha=.1, label=f'MG2{B} 95% CI')
            title = f'E[N] w.r.t. loads. Comparison MM2{B}/MG2{B}'

        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()

def waiting_line_comparison(input_files, out_filename, _type, exp=True):
    
    if _type == 'users':
        for data in input_files:
            B = list(data)[13] 
            if list(data)[14] != '.':
                B += '0'

            loads1, _, mean1, _, _ = pick_avg_cust_values(data) 
            if exp:
                plt.plot(loads1, mean1, label=f'M/M/1/{B} simulation')
            else:
                plt.plot(loads1, mean1, label=f'M/G/1/{B} simulation')

        plt.xlabel('Loads [%]')
        plt.ylabel('Avg customers in the queue')
        if exp:        
            title = 'Effect of finite waiting line on E[N] - exp services'
        else:
            title = 'Effect of finite waiting line on E[N] - uni services'
        plt.title('mean of inter-arrival time fixed at 5s')
        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()

    elif _type == 'loss':
        for data in input_files:
            B = list(data)[13] 
            if list(data)[14] != '.':
                B += '0'

            loads1, _, mean1, _, _ = pick_prob_loss_values(data) 
            if exp:
                plt.plot(loads1, mean1, label=f'M/M/1/{B} simulation')
            else:
                plt.plot(loads1, mean1, label=f'M/G/1/{B} simulation')

        plt.xlabel('Loads [%]')
        plt.ylabel('Loss probability [%]')        
        if exp:        
            title = 'Effect of finite waiting line on loss - exp services'
        else:
            title = 'Effect of finite waiting line on loss - uni services'
        plt.title('mean of inter-arrival time fixed at 5s')
        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()

def multi_comparison(input_files, out_filename, _type):

    if _type == 'delay':
        loads1, lb1, mean1, ub1, _ = pick_delay_values(input_files[0]) 
        plt.plot(loads1, mean1, label='M/M/1 simulation')
        #plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label='M/M/1/2 95% CI')
        loads1, lb1, mean1, ub1, _ = pick_delay_values(input_files[1])
        plt.plot(loads1, mean1, label='M/G/1 simulation')
        #plt.fill_between(loads1, lb1, ub1, color='orange', alpha=.1, label='M/G/1/2 95% CI')
        loads1, lb1, mean1, ub1, _ = pick_delay_values(input_files[2])
        plt.plot(loads1, mean1, label='M/M/2 simulation')
        #plt.fill_between(loads1, lb1, ub1, color='g', alpha=.1, label='M/M/2/2 95% CI')
        loads1, lb1, mean1, ub1, _ = pick_delay_values(input_files[3])
        plt.plot(loads1, mean1, label='M/G/2 simulation')
        #plt.fill_between(loads1, lb1, ub1, color='r', alpha=.1, label='M/G/2/2 95% CI')        

        plt.xlabel('Loads [%]')
        plt.ylabel('Delay [s]')        
        title = 'Effect of multi server with same capacity on delay'        
        plt.title('mean of inter-arrival time fixed at 5s')
        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()

    elif _type == 'loss':
        # files order as MM1b MG1b MM2b MG2b
        loads1, lb1, mean1, ub1, _ = pick_prob_loss_values(input_files[0]) 
        plt.plot(loads1, mean1, label='M/M/1/2 simulation')
        plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label='M/M/1/2 95% CI')
        loads1, lb1, mean1, ub1, _ = pick_prob_loss_values(input_files[1])
        plt.plot(loads1, mean1, label='M/G/1/2 simulation')
        plt.fill_between(loads1, lb1, ub1, color='orange', alpha=.1, label='M/G/1/2 95% CI')
        loads1, lb1, mean1, ub1, _ = pick_prob_loss_values(input_files[2])
        plt.plot(loads1, mean1, label='M/M/2/2 simulation')
        plt.fill_between(loads1, lb1, ub1, color='g', alpha=.1, label='M/M/2/2 95% CI')
        loads1, lb1, mean1, ub1, _ = pick_prob_loss_values(input_files[3])
        plt.plot(loads1, mean1, label='M/G/2/2 simulation')
        plt.fill_between(loads1, lb1, ub1, color='r', alpha=.1, label='M/G/2/2 95% CI')        

        plt.xlabel('Loads [%]')
        plt.ylabel('Loss probability [%]')        
        title = 'Effect of multi server with same capacity on loss'        
        plt.title('mean of inter-arrival time fixed at 5s')
        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()

    elif _type == 'users':
        # files order as MM1b MG1b MM2b MG2b
        loads1, lb1, mean1, ub1, _ = pick_avg_cust_values(input_files[0]) 
        plt.plot(loads1, mean1, label='M/M/1 simulation')
        #plt.fill_between(loads1, lb1, ub1, color='b', alpha=.1, label='M/M/1/2 95% CI')
        loads1, lb1, mean1, ub1, _ = pick_avg_cust_values(input_files[1])
        plt.plot(loads1, mean1, label='M/G/1 simulation')
        #plt.fill_between(loads1, lb1, ub1, color='orange', alpha=.1, label='M/G/1/2 95% CI')
        loads1, lb1, mean1, ub1, _ = pick_avg_cust_values(input_files[2])
        plt.plot(loads1, mean1, label='M/M/2 simulation')
        #plt.fill_between(loads1, lb1, ub1, color='g', alpha=.1, label='M/M/2/2 95% CI')
        loads1, lb1, mean1, ub1, _ = pick_avg_cust_values(input_files[3])
        plt.plot(loads1, mean1, label='M/G/2 simulation')
        #plt.fill_between(loads1, lb1, ub1, color='r', alpha=.1, label='M/G/2/2 95% CI')        

        plt.xlabel('Loads [%]')
        plt.ylabel('Average users in the queue')        
        title = 'Effect of multi server with same capacity E[N]'        
        plt.title('mean of inter-arrival time fixed at 5s')
        plt.suptitle(title)
        plt.legend()
        plt.savefig('Lab1/images/'+out_filename)
        plt.clf()



#metric_wrt_mu_service('Lab1/data/prova.dat', 'exp_delay_wrt_mu_MM1.png', 'delay')
metric_wrt_mu_service('Lab1/data/MM1.dat', 'comparison_delay_Mx1.png', 'delay', comparison='Lab1/data/MG1.dat')
metric_wrt_mu_service('Lab1/data/MM1.dat', 'comparison_ET_Mx1.png', 'avg_users', comparison='Lab1/data/MG1.dat')

# waiting line effect
file_list_exp = ['Lab1/data/MM12.dat', 'Lab1/data/MM15.dat', 'Lab1/data/MM17.dat', 'Lab1/data/MM110.dat', 'Lab1/data/MM120.dat']
file_list_uni = ['Lab1/data/MG12.dat', 'Lab1/data/MG15.dat', 'Lab1/data/MG17.dat', 'Lab1/data/MG110.dat', 'Lab1/data/MG120.dat']
waiting_line_comparison(file_list_exp, 'MM1x_waiting_effect.png', 'users')
waiting_line_comparison(file_list_uni, 'MG1x_waiting_effect.png', 'users', exp=False)
waiting_line_comparison(file_list_exp, 'loss_MM1x_waiting_effect.png', 'loss')
waiting_line_comparison(file_list_uni, 'loss_MG1x_waiting_effect.png', 'loss', exp=False)

# multi server
#mx1b_metrics('Lab1/data/MM2.dat', 'comparison_delay_Mx2.png', 'delay', comparison='Lab1/data/MG2.dat')
#mx1b_metrics('Lab1/data/MM2.dat', 'comparison_ET_Mx2.png', 'avg_users', comparison='Lab1/data/MG2.dat')
#metric_wrt_mu_service('Lab1/data/MM22.dat', 'comparison_loss_Mx22.png', 'loss_probs', comparison='Lab1/data/MG22.dat')
same_multi_files = ['Lab1/data/MM12.dat', 'Lab1/data/MG12.dat', 'Lab1/data/MM22.dat', 'Lab1/data/MG22.dat']
same_multi_files_2 = ['Lab1/data/MM1.dat', 'Lab1/data/MG1.dat', 'Lab1/data/MM2.dat', 'Lab1/data/MG2.dat']
multi_comparison(same_multi_files, 'multi_same_mu_loss.png', 'loss')
multi_comparison(same_multi_files_2, 'multi_same_mu_delay.png', 'delay')
multi_comparison(same_multi_files_2, 'multi_same_mu_users.png', 'users')


