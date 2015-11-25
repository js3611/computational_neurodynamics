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

T  = 1000 * 60  # Simulation time
Ib = 15.0    # extra current to be injected
nRuns = 20

for i in range(nRuns):
  p = rand.random()
  net = create_izModularNetwork(p)

  ## Initialise layers
  for lr in xrange(len(net.layer)):
    net.layer[lr].v = -65 * np.ones(net.layer[lr].N)
    net.layer[lr].u = net.layer[lr].b * net.layer[lr].v
    net.layer[lr].firings = np.array([])

  ## SIMULATE
  for t in xrange(T):

    # Deliver random background current to exhibitory layers
    for i in xrange(len(net.layer)-1):
      net.layer[i].I = Ib * rand.poisson(0.01, net.layer[i].N)
    net.layer[len(net.layer)-1].I = np.zeros(200)  # inhibitory layer

    net.Update(t)
    # print t*1.0/(T*nRuns)*100  # progress

  ## mean firing rate

  window = 50
  shift = 20
  step = 10
  start_time = 1000

  bins = xrange(start_time, T, step)
  time_stamps = xrange(start_time, T, shift)
  mfr = np.zeros([len(time_stamps), 9])
  mfr[:,0] = time_stamps

  for i in xrange(len(net.layer)-1):
    firings = net.layer[i].firings
    aggr_firings = np.array([]).reshape(0,2)
    counts, bins_result = np.histogram(firings[:,0], bins)

    for j in range(len(time_stamps)):
      t0 = time_stamps[j]
      start = (t0-start_time)/step
      end = min(start + window/step, len(counts))
      size = (end-start)*step
      mfr[j,i+1] = np.sum(counts[start:end])*1.0/size

  print mfr[1]

  ## calling Java

  miCalc.initialise(8)
  # miCalc.startAddObservations()
  # miCalc.finaliseAddObservations()
  miCalc.setObservations(mfr[:,range(1,9)])
  result = miCalc.computeAverageLocalOfObservations()*np.log2(np.e)

  print p, result
