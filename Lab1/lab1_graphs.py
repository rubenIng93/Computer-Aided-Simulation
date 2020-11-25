import matplotlib.pyplot as plt
import csv
import os

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

    for file in files_paths:
        plot_graph("Lab1/"+file, file.split(".")[0]+".png")

# if you want to plot all the charts in a single run
get_all_dir_graphs()

