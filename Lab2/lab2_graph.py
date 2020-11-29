import matplotlib.pyplot as plt
import csv
import os

# THIS SCRIPT ASSUMES THAT THE FILE ARE IN A FOLDER CALLED "Lab1"

def plot_graph_probability(input_filename,output_filename):

    m = []
    theory = []
    simulation = []
    ci1=[]
    ci2=[]

    # make a folder
    if os.listdir(os.getcwd()+"/Lab2").__contains__("images") == False:
        os.mkdir(os.getcwd()+"/Lab2/images")

    with open(input_filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        first_line=True
        for row in plots:
            if first_line:
                first_line=False
            else:
                m.append(int(row[0]))
                theory.append(float(row[1])*100)
                simulation.append(float(row[2])*100)
                ci1.append(float(row[3])*100)
                ci2.append(float(row[4])*100)
            
    #plt.vlines(380, linestyles='dashed', label='Conflict threshold:', colors='r', ymin=0, ymax=100)
    plt.plot(m, simulation, label='Simulation', marker='.')
    plt.plot(m,theory, label='Theoretical',linestyle='dotted')
    plt.fill_between(m, ci1,ci2, color='b', alpha=.1, label='95% CI')
    plt.xlabel('Number of people')
    plt.ylabel('Probability of conflicts [%]')
    title = "Simulation vs Theoretical"
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig("Lab2/images/"+output_filename)
    plt.clf()

'''
def plot_graph_min_conflicts_per run(input_filename,output_filename):

    m = []
    simulation = []
    ci1=[]
    ci2=[]

    # make a folder
    if os.listdir(os.getcwd()+"/Lab2").__contains__("images") == False:
        os.mkdir(os.getcwd()+"/Lab2/images")

    with open(input_filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        first_line=True
        for row in plots:
            if first_line:
                first_line=False
            else:
                m.append(int(row[0]))
                theory.append(float(row[1])*100)
                simulation.append(float(row[2])*100)
                ci1.append(float(row[3])*100)
                ci2.append(float(row[4])*100)
            
    #plt.vlines(380, linestyles='dashed', label='Conflict threshold:', colors='r', ymin=0, ymax=100)
    plt.plot(m, simulation, label='Simulation', marker='.')
    plt.plot(m,theory, label='Theoretical',linestyle='dotted')
    plt.fill_between(m, ci1,ci2, color='b', alpha=.1, label='95% CI')
    plt.xlabel('Number of people')
    plt.ylabel('Probability of conflicts [%]')
    title = "Simulation vs Theoretical"
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig("Lab2/images/"+output_filename)
    plt.clf()
'''





plot_graph_probability("Lab2/birthdayparadox100000elements1000runs.dat", "Graph10e5_1000runs.png")