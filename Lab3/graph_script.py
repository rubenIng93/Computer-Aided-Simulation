import matplotlib.pyplot as plt
import csv
import os

def plot_graph_probability(input_filename,output_filename):

    b = []
    analytical = []
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
                analytical.append(float(row[1]))
                ci1.append(float(row[2]))
                mean.append(float(row[3]))
                ci2.append(float(row[4]))
            
    plt.plot(b, mean, label='Mean P(FP)', marker='.')
    plt.plot(b, analytical, label='analytical approach', marker='.')
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
                size.append(float(row[5]))
            
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

def plot_graph_probability_bloom(input_filename,output_filename):

    b = []
    mean = []
    ci1=[]
    ci2=[]
    theory = []

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
                ci1.append(float(row[7]))
                mean.append(float(row[8]))
                ci2.append(float(row[9]))
                theory.append(float(row[10]))
            
    plt.plot(b, mean, label='Mean P(FP)', marker='.')
    plt.plot(b, theory, label='Theoretical P(FP)', marker='.')
    plt.fill_between(b, ci1, ci2, color='b', alpha=.1, label='95% CI')
    plt.xlabel('Number of bit used for the bit-string')
    plt.ylabel('Probability of False Positive [%]')
    plt.grid()
    title = "Probability of false positive in a Bloom Filter"
    plt.title('When storing 370103 english words')
    plt.suptitle(title)
    plt.legend()
    # plt.show()
    plt.savefig("Lab3/images/"+output_filename)
    plt.clf()

def plot_kopt_vs_bits(input_filename,output_filename):

    b = []
    k_opt = []
    
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
                k_opt.append(float(row[6]))                
            
    plt.plot(b, k_opt, marker='.')
    plt.xlabel('Number of bit used for the bit-string')
    plt.ylabel('Number of optimal hash functions')
    plt.grid()
    title = "Optimal number of hash function to minimize P(FP)"
    plt.title('When storing 370103 english words')
    plt.suptitle(title)
    # plt.show()
    plt.savefig("Lab3/images/"+output_filename)
    plt.clf()




plot_graph_probability('Lab3/lab3100runs.dat', 'probability_chart.png')
plot_graph_probability_bloom('Lab3/lab3100runs.dat', 'bf_prob_chart.png')
plot_size_graph('Lab3/lab3100runs.dat', 'size_chart.png')
plot_kopt_vs_bits('Lab3/lab3100runs.dat', 'k_opt_chart.png')