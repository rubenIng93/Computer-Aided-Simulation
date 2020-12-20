import matplotlib.pyplot as plt
import csv
import os

def plot_graph_probability(input_filename,output_filename):

    b = []
    mean = []
    ci1=[]
    ci2=[]

    # make a folder
    if os.listdir(os.getcwd()+"/Lab3").__contains__("images") == False:
        os.mkdir(os.getcwd()+"/Lab3/images")

    with open(input_filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        first_line=True
        for row in plots:
            if first_line:
                first_line=False
            else:
                b.append(int(row[0]))
                ci1.append(float(row[1]))
                mean.append(float(row[2]))
                ci2.append(float(row[3]))
            
    plt.plot(b, mean, label='Mean P(FP)', marker='.')
    plt.fill_between(b, ci1, ci2, color='b', alpha=.1, label='95% CI')
    plt.xlabel('Number of bit used for the bit-string')
    plt.ylabel('Probability of False Positive [%]')
    plt.grid()
    title = "Probability of false positive in a bit-string array"
    plt.title('When storing 370103 english words')
    plt.suptitle(title)
    plt.legend()
    # plt.show()
    plt.savefig("Lab3/images/"+output_filename)
    plt.clf()

def plot_size_graph(input_filename,output_filename):

    b = []
    size = []

    # make a folder
    if os.listdir(os.getcwd()+"/Lab3").__contains__("images") == False:
        os.mkdir(os.getcwd()+"/Lab3/images")

    with open(input_filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        first_line=True
        for row in plots:
            if first_line:
                first_line=False
            else:
                b.append(int(row[0]))
                size.append(float(row[4]))
            
    plt.plot(b, size, marker='.')
    plt.xlabel('Number of bit used for the bit-string')
    plt.ylabel('Actual storage size [kb]')
    plt.grid()
    title = "Actual storage vs #bits used"
    plt.title('When storing 370103 english words')
    plt.suptitle(title)
    # plt.show()
    plt.savefig("Lab3/images/"+output_filename)
    plt.clf()

plot_graph_probability('Lab3/BitstringHash100runs.dat', 'probability_chart.png')
plot_size_graph('Lab3/BitstringHash100runs.dat', 'size_chart.png')