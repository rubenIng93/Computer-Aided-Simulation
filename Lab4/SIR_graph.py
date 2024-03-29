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

def sir_chart_ci(input_filename,output_filename, extension=''):

    days = []
    mean_s = []
    cil_s = []
    cih_s = []
    mean_i = []
    cil_i = []
    cih_i = []
    mean_r = []
    cil_r = []
    cih_r = []
    
    with open(input_filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        first_line=True
        for row in plots:
            if first_line:
                first_line=False
            else:
                days.append(int(row[0]))
                mean_s.append(float(row[1]))
                cil = float(row[2])
                if cil >= 0:
                    cil_s.append(cil)
                else:
                    cil_s.append(0)
                cih_s.append(float(row[3]))
                mean_i.append(float(row[5]))
                cil = float(row[6])
                if cil >= 0:
                    cil_i.append(cil)
                else:
                    cil_i.append(0)
                cih_i.append(float(row[7]))
                mean_r.append(float(row[9]))
                cil = float(row[10])
                if cil >= 0:
                    cil_r.append(cil)
                else:
                    cil_r.append(0)
                cih_r.append(float(row[11]))
            
    plt.plot(days, mean_s, label='S(t)')
    plt.fill_between(days, cil_s, cih_s, color='b', alpha=.1, label='95% CI')
    plt.plot(days, mean_i, label='I(t)')
    plt.fill_between(days, cil_i, cih_i, color='r', alpha=.1, label='95% CI')
    plt.plot(days, mean_r, label='R(t)')
    plt.fill_between(days, cil_r, cih_r, color='g', alpha=.1, label='95% CI')
    #plt.plot(days, i_t, label='I(t)')
    #plt.plot(days, r_t, label='R(t)')

    plt.xlabel('Days')
    plt.ylabel('Number of peoples')
    plt.grid()
    plt.suptitle('Disease Trend - Simulative solution')
    plt.title(extension)
    plt.legend()
    # plt.show()
    plt.savefig("Lab4/images/"+output_filename)
    plt.clf()


def plot_rt_trend(input_filename,output_filename, extension=''):

    days = []
    mean = []
    cil = []
    cih = []
    
    
    with open(input_filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        first_line=True
        for row in plots:
            if first_line:
                first_line=False
            else:
                days.append(int(row[0]))
                mean.append(float(row[13]))
                low = float(row[14])
                if low >= 0:
                    cil.append(low)
                else:
                    cil.append(0)
                cih.append(float(row[15]))
                
            
    plt.plot(days, mean, label='Rt (avg)')
    plt.fill_between(days, cil, cih, color='b', alpha=.1, label='95% CI')
    
    #plt.plot(days, i_t, label='I(t)')
    #plt.plot(days, r_t, label='R(t)')

    plt.xlabel('Days')
    plt.ylabel('Rt index')
    plt.grid()
    plt.suptitle('Rt Trend - Simulative solution')
    plt.title(extension)
    plt.legend()
    # plt.show()
    plt.savefig("Lab4/images/"+output_filename)
    plt.clf()


#plot_graph_SIR('Lab4/SIRmodel_numerical.dat', 'analytical_SIR.png')
#sir_chart_ci('Lab4/simulative_SIR.dat', 'simulative_SIR.png')
#plot_rt_trend('Lab4/simulative_SIR.dat', 'rt_chart.png')
sir_chart_ci('Lab4/SIR_ext.dat', 'simulative_SIR_ext.png', 'without mask')
plot_rt_trend('Lab4/SIR_ext.dat', 'rt_chart_nomask.png', 'without mask')
sir_chart_ci('Lab4/simulative_SIRS.dat', 'SIRS_model.png', 'SIRS extension')
plot_rt_trend('Lab4/simulative_SIRS.dat', 'rt_chart_sirs.png', 'SIRS extension')
sir_chart_ci('Lab4/simulative_SIRS_lock.dat', 'SIRS_model_lock.png', 'SIRS extension with Lockdown')
plot_rt_trend('Lab4/simulative_SIRS_lock.dat', 'rt_chart_sirs_lock.png', 'SIRS extension with Lockdown')
