import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv

def plot_comparison_ex1(exp_list, uni_list, out_filename, delays=False):

    labels = ['40', '60', '80', '90', '98']
    x = np.arange(len(labels)) # the label location
    width = 0.35 # width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, exp_list, width, label='Exponential')
    rects2 = ax.bar(x + width/2, uni_list, width, label='Uniform')

    ax.set_xlabel('Load [%]')
    if delays:
        ax.set_title('E[T] by loads and service time distribution')
        ax.set_ylabel('Delays [s]')
    else:
        ax.set_title('E[N] by loads and service time distribution')
        ax.set_ylabel('Avg users')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig('Lab1/images/'+out_filename)



# find the .csv files
exp_files_paths = []
uni_files_paths = []
for _file in os.listdir(os.getcwd()+"/Lab1"):
    if _file.startswith("exp"):
        exp_files_paths.append('Lab1/'+_file)
    if _file.startswith("uni"):
        uni_files_paths.append('Lab1/'+_file)

# make a folder
if os.listdir(os.getcwd()+"/Lab1").__contains__("images") == False:
    os.mkdir(os.getcwd()+"/Lab1/images")

# load the files and save the useful variables
# i.e. E[N] and E[T]
uni_users = []
uni_delays = []
exp_users = []
exp_delays = []

# LOOP THAT GETS THE RIGHT VALUES FOR EXP SERVICES
for _file in exp_files_paths:
    with open(_file, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        i = 0
        for row in plots:
            if i == 2:
                exp_users.append(float(row[3])) # avg users
            elif i == 5:
                exp_delays.append(float(row[3])) # delays
            i += 1

# LOOP THAT GETS THE RIGHT VALUES FOR UNIFORM SERVICES
for _file in uni_files_paths:
    with open(_file, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        i = 0
        for row in plots:
            if i == 2:
                uni_users.append(float(row[3])) # avg users
            elif i == 5:
                uni_delays.append(float(row[3])) # delays
            i += 1



plot_comparison_ex1(exp_users, uni_users, 'avg_users_comparison.png')
plot_comparison_ex1(exp_delays, uni_delays, 'delays_comparison.png')


