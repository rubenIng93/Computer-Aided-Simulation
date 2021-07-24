import matplotlib.pyplot as plt
import csv
import os

def plot_graph_SIR(input_filename,output_filename):

    days = []
    s_t = []
    i_t = []
    r_t = []

    # make a folder
    if os.listdir(os.getcwd()+"/Lab4").__contains__("images") == False:
        os.mkdir(os.getcwd()+"/Lab4/images")

    with open(input_filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        first_line=True
        for row in plots:
            if first_line:
                first_line=False
            else:
                days.append(int(row[0]))
                s_t.append(int(row[1]))
                i_t.append(int(row[2]))
                r_t.append(int(row[3]))
            
    plt.plot(days, s_t, label='S(t)')
    plt.plot(days, i_t, label='I(t)')
    plt.plot(days, r_t, label='R(t)')

    plt.xlabel('Days')
    plt.ylabel('Number of peoples')
    plt.grid()
    plt.title('Disease Trend - Numerical solution')
    plt.legend()
    # plt.show()
    plt.savefig("Lab4/images/"+output_filename)
    plt.clf()

plot_graph_SIR('Lab4/SIRmodel_numerical.dat', 'analytical_SIR.png')
plot_graph_SIR('Lab4/SIRmodel_agentBased2.dat', 'AB_SIR2.png')