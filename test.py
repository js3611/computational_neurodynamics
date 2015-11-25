__author__ = 'js3611'

from IzModularNetwork import create_izModularNetwork
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt



############# Q2 #############


T  = 1000 * 5  # Simulation time
Ib = 15.0    # extra current to be injected

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
  net.layer[8].I = np.zeros(200)

  net.Update(t)

## mean firing rate

window = 50
shift = 20
step = 10
start_time = 1000

mfr = []
for i in xrange(len(net.layer)-1):
  firings = net.layer[i].firings
  aggr_firings = np.array([]).reshape(0,2)
  counts, bins = np.histogram(firings[:,0], xrange(0,T,step))

  for t0 in range(start_time, T, shift):
    start = t0/step
    end = min(start + window/step, len(counts))
    size = (end-start)*step
    aggr_firings = np.vstack((aggr_firings, [t0, np.sum(counts[start:end])*1.0/size]))

  mfr.append(aggr_firings)

print mfr[1]

for i in xrange(len(net.layer)-1):
  plt.plot(mfr[i][:,0], mfr[i][:,1])
plt.title('Mean firing rate')

plt.show()
