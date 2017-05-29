# -*- coding: utf-8 -*-
"""
Created on Mon May 29 08:28:26 2017
Demonstration for using the fmri-sim module

@author: rmillin
"""


# import modules

import fmriSim as fmri
import matplotlib.pyplot as plt
import numpy as np


# parameters

timing = np.linspace( 0, 199, 200 )
TE = 0.03;
protocol = np.ones(200)
#plotdata = False;
for x in range(0, int(200/20)-1):
    protocol[x*20+11:x*20+15-1] = 1.7

protocol = protocol-1
deltat = 1
const = 1
    
    
# run it and plot

# stimulus to neural response
tneur, neur = fmri.stimulusToNeural(protocol,deltat,const)
timing = [deltat*i for i in range(len(protocol))] # time vector
# Stim and neural response
fig1 = plt.figure(figsize=(12,4))
# Plot the stimulus
ax1 = fig1.add_subplot(121)
ax1.plot(timing, protocol)
ax1.set_xlabel('time')
ax1.set_ylabel('stimulus')
# Plot the neural response
ax2 = fig1.add_subplot(122)
ax2.plot(tneur, neur)
ax2.set_xlabel('time')
ax2.set_ylabel('neural response')
plt.show()    

# neural to flow in
tflowin, flowin, vascsignal = fmri.neuralToFlow(tneur,neur)
# neurovascular signal and flow
fig2 = plt.figure(figsize=(12,4))
# Plot the signal to the vasculature
ax1 = fig2.add_subplot(121)
ax1.plot(tflowin, vascsignal)
ax1.set_xlabel('time')
ax1.set_ylabel('signal to vasculature')
# Plot the neural response
ax2 = fig2.add_subplot(122)
ax2.plot(tflowin, flowin)
ax2.set_xlabel('time')
ax2.set_ylabel('flow in to vasculature')
plt.show()



# flow in to BOLD
t, resp, q, v = fmri.balloonModel(tflowin,flowin,TE)
# Balloon model plots
fig3 = plt.figure(figsize=(18,4))
# Plot dexoyhem conc as a function of time
ax1 = fig3.add_subplot(131)
ax1.plot(t, q)
ax1.set_xlabel('time')
ax1.set_ylabel('[deoxyhemoglobin]')
# Plot volume as a function of time
ax2 = fig3.add_subplot(132)
ax2.plot(t, v)
ax2.set_xlabel('time')
ax2.set_ylabel('blood volume')
# Plot BOLD as a function of time
ax3 = fig3.add_subplot(133)
ax3.plot(t, resp)
ax3.set_xlabel('time')
ax3.set_ylabel('BOLD')
# plt.tight_layout()
plt.show()

