__author__ = 'js3611'

from IzModularNetwork import create_izModularNetwork
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from jpype import *


############# Q2 #############

jarLocation = "infodynamics-dist-1.3/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

miCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov2
miCalc = miCalcClass()

n_ex_layer=8
N_ex=100
N_in=200
T  = 1000*60  # Simulation time
nRuns = 100  # number of runs
Ib = 15.0    # extra current to be injected
window = 50  # mean firing rate averaging window
shift = 20  # mean firing rate shift
step = 10  # bin size in histogram
start_time = 1000  # starting time for counting mean firing rate


data = np.array([]).reshape(0,2)
for run in range(nRuns):
  p = rand.random()  # random rewiring probability
  net = create_izModularNetwork(p)

  # Initialise layers
  for lr in xrange(len(net.layer)):
    net.layer[lr].v = -65 * np.ones(net.layer[lr].N)
    net.layer[lr].u = net.layer[lr].b * net.layer[lr].v
    net.layer[lr].firings = np.array([])

  # Simulate
  for t in xrange(T):
    # Deliver random background current to exhibitory layers
    for i in xrange(len(net.layer)-1):
      net.layer[i].I = Ib * rand.poisson(0.01, net.layer[i].N)

    # net.layer[n_ex_layer].I = np.zeros(N_in)  # inhibitory layer
    net.layer[n_ex_layer].I = Ib * rand.poisson(0.01, net.layer[n_ex_layer].N)  # also injecting background current to inhibitory layer
    net.Update(t)

  # Calculating mean firing rate

  bins = xrange(start_time, T, step)
  time_stamps = xrange(start_time, T, shift)
  mfr = np.zeros([len(time_stamps), N_ex+1])  # mean firing rate
  mfr[:,0] = time_stamps  # first index is time stamp

  for i in xrange(len(net.layer)-1):
    firings = net.layer[i].firings
    counts, bins_result = np.histogram(firings[:,0], bins)

    for j in range(len(time_stamps)):
      t0 = time_stamps[j]
      start = (t0-start_time)/step
      end = min(start + window/step, len(counts))
      size = (end-start)*step
      mfr[j,i+1] = np.sum(counts[start:end])*1.0/size

  # Calling Java library

  miCalc.initialise(n_ex_layer)
  miCalc.setObservations(mfr[:,1:n_ex_layer+1])
  result = miCalc.computeAverageLocalOfObservations()*np.log2(np.e)  # convert form nats to bits
  # Append result 
  # print run, p, result
  data = np.vstack((data, [p, result]))

# Plot Result
plt.scatter(data[:,0], data[:,1], marker='.')
plt.xlim((0,1))
plt.ylim((0,5))
plt.title('Integration I(S)')
plt.xlabel('Rewiring probability p')
plt.ylabel('Integration (bits)')
# plt.plot()
plt.savefig('Multi-Information of mean firing rates.svg')
