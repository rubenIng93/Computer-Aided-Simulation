import matplotlib.pyplot as plt
import csv
import os

# THIS SCRIPT ASSUMES THAT THE FILE ARE IN A FOLDER CALLED "Lab1"

def plot_graph(input_filename,output_filename):

    n = []
    ave = []
    lower_bound=[]
    upper_bound=[]
    ci1=[]
    ci2=[]
    flag = False

    with open(input_filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        first_line=True
        for row in plots:
            if first_line:
                first_line=False
            else:
                if len(row) == 7:
                    n.append(float(row[0]))
                    lower_bound.append(float(row[1]))
                    upper_bound.append(float(row[2]))
                    ci1.append(float(row[3]))
                    ave.append(float(row[4]))
                    ci2.append(float(row[5]))
                else:
                    n.append(float(row[0]))
                    lower_bound.append(float(row[1]))
                    ci1.append(float(row[2]))
                    ave.append(float(row[3]))
                    ci2.append(float(row[4]))
                    flag = True

    if flag == False:
        plt.plot(n, upper_bound, label='Upper bound', linestyle='dotted')
        plt.plot(n,lower_bound, label='Lower bound',linestyle='dotted')

    plt.plot(n,ave, label='Simulation',marker='o')

    if flag:
        plt.plot(n,lower_bound, label='Theory',linestyle='dotted')

    plt.xscale("log")
    plt.fill_between(n, ci1,ci2, color='b', alpha=.1, label='95% CI')
    plt.xlabel('Bins')
    plt.ylabel('Max bin occupancy')
    if flag:
        plt.ylim(bottom=0, top=10)
    else:
        plt.ylim(bottom=0)
    title = input_filename.split(".")[0]
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig(output_filename)
    plt.clf()

def get_all_dir_graphs():
    # find the .dat files
    files_paths = []
    for file in os.listdir(os.getcwd()+"/Lab1"):
        if file.find(".dat") > 0:
            files_paths.append(file)
    
    # make a folder
    if os.listdir(os.getcwd()+"/Lab1").__contains__("images") == False:
        os.mkdir(os.getcwd()+"/Lab1/images")

    for file in files_paths:
        plot_graph("Lab1/"+file, "Lab1/images/"+file.split(".")[0]+".png")

def comparison_graph(mode):
    # mode can be relative errors or max occupancy
    n = []
    if mode == 'max_occupancy':
        # this branch compares among different policies
        ave = []
        ci1 = []
        ci2 = []
        # random dropping
        with open("Lab1/binsballs10runs.dat",'r') as csvfile:
            plots = csv.reader(csvfile, delimiter='\t')
            first_line=True
            for row in plots:
                if first_line:
                    first_line=False
                else:
                    n.append(float(row[0]))
                    ci1.append(float(row[3]))
                    ave.append(float(row[4]))
                    ci2.append(float(row[5]))
        plt.plot(n,ave, label='Random dropping policy',marker='o')
        plt.xscale("log")
        plt.fill_between(n, ci1,ci2, color='b', alpha=.1, label='95% CI')
        plt.xlabel('Bins')
        plt.ylabel('Max bin occupancy')
        # load balancing 2
        ave = []
        ci1 = []
        ci2 = []
        with open("Lab1/binsballs10runs_load_balancing2.dat",'r') as csvfile:
            plots = csv.reader(csvfile, delimiter='\t')
            first_line=True
            for row in plots:
                if first_line:
                    first_line=False
                else:
                    ci1.append(float(row[2]))
                    ave.append(float(row[3]))
                    ci2.append(float(row[4]))

        plt.plot(n,ave, label='Random load balancing = 2',marker='X')
        plt.fill_between(n, ci1,ci2, color='r', alpha=.1, label='95% CI')
        # load balancing 4
        ave = []
        ci1 = []
        ci2 = []
        with open("Lab1/binsballs10runs_load_balancing4.dat",'r') as csvfile:
            plots = csv.reader(csvfile, delimiter='\t')
            first_line=True
            for row in plots:
                if first_line:
                    first_line=False
                else:
                    ci1.append(float(row[2]))
                    ave.append(float(row[3]))
                    ci2.append(float(row[4]))

        plt.plot(n,ave, label='Random load balancing = 4',marker='D')
        plt.fill_between(n, ci1,ci2, color='g', alpha=.1, label='95% CI')  
        plt.ylim(bottom=0, top=10)
        plt.legend()
        plt.title("Comparison between policies")
        plt.savefig("Lab1/images/Max_occupancy_comparison.png")
        plt.clf() 

    elif mode == 'rel_error':
        # this branch compares among different num of runs for the random dropping policy
        rel_err = []
        # 5 runs
        with open("Lab1/binsballs5runs.dat",'r') as csvfile:
            plots = csv.reader(csvfile, delimiter='\t')
            first_line=True
            for row in plots:
                if first_line:
                    first_line=False
                else:
                    n.append(float(row[0]))
                    rel_err.append(float(row[6])*100)
        plt.plot(n,rel_err, label='5 Runs',marker='.')
        plt.xscale("log")
        plt.xlabel('Bins')
        plt.ylabel('Relative error %')
        # 10 runs
        rel_err = []
        with open("Lab1/binsballs10runs.dat",'r') as csvfile:
            plots = csv.reader(csvfile, delimiter='\t')
            first_line=True
            for row in plots:
                if first_line:
                    first_line=False
                else:
                    rel_err.append(float(row[6])*100)

        plt.plot(n,rel_err, label='10 Runs',marker='.')
        # 40 runs
        rel_err = []
        with open("Lab1/binsballs40runs.dat",'r') as csvfile:
            plots = csv.reader(csvfile, delimiter='\t')
            first_line=True
            for row in plots:
                if first_line:
                    first_line=False
                else:
                    rel_err.append(float(row[6])*100)

        plt.plot(n,rel_err, label='40 Runs',marker='.')
        plt.ylim(bottom=0)
        plt.legend()
        plt.title("Comparison between number of runs")
        plt.savefig("Lab1/images/N_runs_comparison.png")
        plt.clf()
    else:
        print('As mode insert "max_occupancy" or "rel_error"')
        return
# if you want to plot all the charts in a single run
#get_all_dir_graphs()
#comparison_graph("max_occupancy")
comparison_graph("prova")