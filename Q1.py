__author__ = 'js3611'

from IzModularNetwork import create_izModularNetwork
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt





# p_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
p_range = [0]

for p in p_range:
    net = create_izModularNetwork(p)

############# Q1 (a) #############

    vertical = {}
    for i in xrange(len(net.layer)):
        vertical[i] = np.concatenate(net.layer[i].S.values(), axis=1)
    network_matrix = np.concatenate(vertical.values(), axis=0)
    plt.matshow(network_matrix, fignum=100, cmap=plt.cm.gray)
    plt.show()


############# Q1(b) #############


T  = 1000  # Simulation time
Ib = 15.0    # extra current to be injected

net = create_izModularNetwork(0.2)

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



## Retrieve firings and add Dirac pulses for presentation
firings = net.layer[0].firings
if len(firings) == 0:
  firings = firings.reshape(0,2)

num = 0  # accumulative number of neurons
for i in xrange(1,len(net.layer)):
  num += net.layer[i-1].N
  firings_i = net.layer[i].firings
  if len(firings_i)>0:
    # number = np.ones(len(firings_i))*num  # adjust neuron indices
    firings_i[:,1] += num
    firings = np.vstack((firings, firings_i))

## Raster plots of firings
if firings.size != 0:
  plt.scatter(firings[:, 0], firings[:, 1] + 1, marker='.', color='blue')
  plt.xlim(0, T)
  plt.ylabel('Neuron number')
  plt.ylim(0, 1000+1)
  plt.title('Population 1 firings')

plt.show()


############# Q1(c) #############

window = 50
shift = 20
step = 10

mfr = []  # mean firing rate
for i in xrange(len(net.layer)-1):
  firings = net.layer[i].firings
  aggr_firings = np.array([]).reshape(0,2)
  counts, bins = np.histogram(firings[:,0], xrange(0,T,step))

  for t0 in range(0, T, shift):
    start = t0/step
    end = min(start + window/step, len(counts))
    size = (end-start)*step
    aggr_firings = np.vstack((aggr_firings, [t0, np.sum(counts[start:end])*1.0/size]))

  mfr.append(aggr_firings)

for i in xrange(len(net.layer)-1):
  plt.plot(mfr[i][:,0], mfr[i][:,1])
plt.title('Mean firing rate')

plt.show()
