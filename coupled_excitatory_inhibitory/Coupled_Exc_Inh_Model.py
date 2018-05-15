# This code implements a 4-population neural model consisting of one excitatory and three inhibitory sub-populations.
# The model is based on the math presented in these papers:
#
# Litwin-Kumar, A., Rosenbaum, R. & Doiron, B. Inhibitory stabilization and visual coding in cortical circuits with
# multiple interneuron subtypes. J. Neurophysiol. 115, 1399–1409 (2016).
#
# and
#
# Kuchibhotla, Kishore V., Jonathan V. Gill, Grace W. Lindsay, Eleni S. Papadoyannis, Rachel E. Field, Tom A. Hindmarsh
# Sten, Kenneth D. Miller, and Robert C. Froemke. "Parallel processing by cortical inhibition enables context-dependent
# behavior." Nature neuroscience 20, no. 1 (2017): 62-71.
#
# The model is extendable to have more than 4 sub-populations
# code by: Masoud Ghodrati, Jan 2018

import numpy as np
import matplotlib.pyplot as plt
import Input_Currents

# Initialization
Stimulation_Time = 1000  # ms, total time of simulation
dt = 1                   # ms, time step/resolution of simulation
Sim_steps = Stimulation_Time/dt
n = 2.2                  # the power-law function that determines the relationship between membrance potential and spik rate

k = 0.01                 # Scaling/gain factor
Exc_N1 = 1               # Number of excitatory neural populations, type 1
Inh_N1 = 1               # Number of inhibitory neural populations, type 1
Inh_N2 = 1               # Number of inhibitory neural populations, type 2
Inh_N3 = 1               # Number of inhibitory neural populations, type 3
N_all = Exc_N1 + Inh_N1 + Inh_N2 + Inh_N3

Tau = [300, 100, 10, 10]  # time constant of every population

Ach_max = 9               # Maximum strength of ach (acetylcholine (ACh) receptor)
Rectification_max = 0     # used for rectification
All_Achs = np.array([0, 1, 1, 1]) * Ach_max  # Acetylcholine (ACh) receptor activation

#  Generating the input current to neural populations
Fs = 1000/dt
Fc = 3                   # hertz
Ml = 0.5
Amp = 1
Noise = 0.2
It, t = Input_Currents.Sqr_current(Stimulation_Time / 1000, Ml, Amp, Fs, Fc, Noise)
It = It[0]

# Weigh matrix between inhibitory and excitatory neural populations. Based on previous studies
Weight_Matrix = np.array([[0.017,  -1.956, -0.045, -0.512],
                          [0.1535, -0.99,  -0.09,  -0.307],
                          [2.104,  -0.184,  0,     -0.734],
                          [1.285,   0,     -0.14,   0]])


plt.close('all')

for tri in [1, 0]:


    print(All_Achs)
    All_Achs = All_Achs * tri       # with or without Ach

    SpontRates = np.array([12.8, 24, 8, 8.9])* np.array([1, 1, 1, 1])       # Setting the spontaneous rate
    Input_Current_Amp = np.array([93, 74, 0, 0])                            # Setting the amplitude of input current

    # Input current to every sub-population
    Input_It_to_Exc_N1 = It * Input_Current_Amp[0]
    Input_It_to_Inh_N1 = It * Input_Current_Amp[1]
    Input_It_to_Inh_N2 = It * Input_Current_Amp[2]
    Input_It_to_Inh_N3 = It * Input_Current_Amp[3]

    # Vectors for storing firing rates
    Rate_Exc_N1 = np.zeros((Exc_N1, len(t)))
    Rate_Inh_N1 = np.zeros((Inh_N1, len(t)))
    Rate_Inh_N2 = np.zeros((Inh_N2, len(t)))
    Rate_Inh_N3 = np.zeros((Inh_N3, len(t)))

    # Simulation
    for tl in range(1, int(Sim_steps)-1):

        d_Rate_Exc_N1 = (-Rate_Exc_N1[0, tl - 1] + k * np.maximum((Input_It_to_Exc_N1[tl - 1] + All_Achs[0] + SpontRates[0] + Weight_Matrix[0, :].dot(np.array([Rate_Exc_N1[0, tl - 1], Rate_Inh_N1[0, tl - 1], Rate_Inh_N2[0, tl - 1], Rate_Inh_N3[0, tl - 1]]).T)), Rectification_max) ** n) / Tau[0]
        d_Rate_Inh_N1 = (-Rate_Inh_N1[0, tl - 1] + k * np.maximum((Input_It_to_Inh_N1[tl - 1] + All_Achs[1] + SpontRates[1] + Weight_Matrix[1, :].dot(np.array([Rate_Exc_N1[0, tl - 1], Rate_Inh_N1[0, tl - 1], Rate_Inh_N2[0, tl - 1], Rate_Inh_N3[0, tl - 1]]).T)), Rectification_max) ** n) / Tau[1]
        d_Rate_Inh_N2 = (-Rate_Inh_N2[0, tl - 1] + k * np.maximum((Input_It_to_Inh_N2[tl - 1] + All_Achs[2] + SpontRates[2] + Weight_Matrix[2, :].dot(np.array([Rate_Exc_N1[0, tl - 1], Rate_Inh_N1[0, tl - 1], Rate_Inh_N2[0, tl - 1], Rate_Inh_N3[0, tl - 1]]).T)), Rectification_max) ** n) / Tau[2]
        d_Rate_Inh_N3 = (-Rate_Inh_N3[0, tl - 1] + k * np.maximum((Input_It_to_Inh_N3[tl - 1] + All_Achs[3] + SpontRates[3] + Weight_Matrix[3, :].dot(np.array([Rate_Exc_N1[0, tl - 1], Rate_Inh_N1[0, tl - 1], Rate_Inh_N2[0, tl - 1], Rate_Inh_N3[0, tl - 1]]).T)), Rectification_max) ** n) / Tau[3]

        Rate_Exc_N1[:, tl] = Rate_Exc_N1[:, tl - 1] + d_Rate_Exc_N1 * dt
        Rate_Inh_N1[:, tl] = Rate_Inh_N1[:, tl - 1] + d_Rate_Inh_N1 * dt
        Rate_Inh_N2[:, tl] = Rate_Inh_N2[:, tl - 1] + d_Rate_Inh_N2 * dt
        Rate_Inh_N3[:, tl] = Rate_Inh_N3[:, tl - 1] + d_Rate_Inh_N3 * dt

    # Ploting the outputs
    print(tri)
    plt.subplot(2,1,tri+1)
    plt.plot(t, It, label = "It")
    plt.plot(t, Rate_Exc_N1[0], label = "FR Exc 1")
    plt.plot(t, Rate_Inh_N1[0], label = "FR Inh 1")
    plt.plot(t, Rate_Inh_N2[0], label = "FR Inh 2")
    plt.plot(t, Rate_Inh_N3[0], label = "FR Inh 3")
    plt.ylabel('Firing Rate')
    if tri:
        plt.title('With Arch')
        plt.xlabel('Time')
    else:
        plt.legend()
        plt.legend(frameon=False)
        plt.title('Without Ach')



plt.show()