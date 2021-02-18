import matplotlib.pyplot as plt
import csv
import os

# THIS SCRIPT ASSUMES THAT THE FILE ARE IN A FOLDER CALLED "Lab1"

def get_n_elements(filename):
    e_chunk = filename.split('e')
    elements = e_chunk[0].split('x')[1]
    return elements

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
    plt.plot(m, simulation, label='Simulation')
    plt.plot(m,theory, label='Theoretical',linestyle='dotted', color='r')
    plt.fill_between(m, ci1,ci2, color='b', alpha=.1, label='95% CI')
    plt.xlabel('Number of people m')
    plt.ylabel('Probability of conflicts [%]')
    n_elements = get_n_elements(input_filename)
    title = f"Simulation vs Theoretical - n = {n_elements}"
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig("Lab2/images/"+output_filename)
    plt.clf()

def plot_graph_min_conflicts_per_run(input_filename,output_filename):

    m = []
    m_conflict = []
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
                m_conflict.append(float(row[7]))
                ci1.append(float(row[6]))
                ci2.append(float(row[8]))
            
    plt.plot(m, m_conflict, marker='.')
    plt.fill_between(m, ci1,ci2, color='b', alpha=.1, label='95% CI')
    plt.xlabel('Number of people')
    plt.ylabel('Minimum peoples for a conflict')
    plt.grid()
    plt.legend()
    title = "Minimum m for a conflict over the number of people"
    plt.title(title)
    plt.savefig("Lab2/images/"+output_filename)
    plt.clf()


plot_graph_probability("Lab2/data/birthdayparadox365elements1000runs.dat", "Graph365_1000runs.png") # 365
plot_graph_probability("Lab2/data/birthdayparadox100000elements1000runs.dat", "Graph10e5_1000runs.png") # 10^5 
plot_graph_probability("Lab2/data/birthdayparadox1000000elements1000runs.dat", "Graph10e6_1000runs.png") # 10^6
#plot_graph_min_conflicts_per_run("data/Lab2/birthdayparadox1000000elements100runs.dat", "Conflicts10e6_1000runs.png")
