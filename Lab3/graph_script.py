import matplotlib.pyplot as plt
import csv
import os

from numpy.core.fromnumeric import argmin

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

def plot_size_graph(input_filename,output_filename, bloom_filter=False):

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
                if bloom_filter:
                    size.append(float(row[11]))
                else:
                    size.append(float(row[5]))
            
    plt.plot(b, size, marker='.')
    plt.xlabel('Number of bit used for the bit-string')
    plt.ylabel('Actual storage size [kb]')
    plt.grid()
    if bloom_filter:
        title = "Actual storage vs #bits used - Bloom Filter"
    else:
        title = "Actual storage vs #bits used - Bit string Array"
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


def plot_optional10(input_filename,output_filename):

    b = []
    fp_prob = {x:[] for x in range(19, 25)}
    k = []
    
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
                bits = int(row[0])
                k_opt = int(row[33])
                b.append(bits)
                k.append(k_opt)
                for hash in range(1,33):  
                    fp_prob[bits].append(float(row[hash]))
    
    # plot the optimal lines
    for bits in fp_prob.keys():
        plt.plot(range(1,33), fp_prob[bits], marker='.', label=f'{bits}-bits')
    for bits in fp_prob.keys():
        plt.plot(argmin(fp_prob[bits])+1, min(fp_prob[bits]), marker='|', label='Min Theory', color='r')
        plt.plot(k[bits-19], fp_prob[bits][k[bits-19]-1], marker='x', label='Min Simulated', color='y')
    
    plt.xlabel('Number of hashes')
    plt.ylabel('Probability of False Positive [%]')
    plt.grid()
    title = "Optimal number of hash functions w.r.t number of bits"
    plt.title('Theory vs Simulation - Bloom Filter')
    plt.suptitle(title)
    plt.legend(['19-bits','20-bits','21-bits','22-bits','23-bits','24-bits','Min Theory','Min Simulated'])
    # plt.show()
    plt.savefig("Lab3/images/"+output_filename)
    plt.clf()

plot_graph_probability('Lab3/lab3100runs.dat', 'probability_chart.png')
plot_graph_probability_bloom('Lab3/lab3100runs.dat', 'bf_prob_chart.png')
plot_kopt_vs_bits('Lab3/lab3100runs.dat', 'k_opt_chart.png')
plot_size_graph('Lab3/lab3100runs.dat', 'size_chart_bitstring.png')
plot_size_graph('Lab3/lab3100runs.dat', 'size_chart_bloomfilter.png', bloom_filter=True)
plot_optional10('Lab3\optional10.dat', 'hash_chart.png')