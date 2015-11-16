__author__ = 'js3611'

from IzModularNetwork import create_izModularNetwork
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt





############# Q1 (a) #############
# p_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
p_range = [0]

for p in p_range:
    net = create_izModularNetwork(p)


    # Print out the matrix for fun
    vertical = {}
    for i in xrange(len(net.layer)):
        vertical[i] = np.concatenate(net.layer[i].S.values(), axis=1)
    network_matrix = np.concatenate(vertical.values(), axis=0)
    plt.matshow(network_matrix, fignum=100, cmap=plt.cm.gray)
    plt.show()


############# Q1(b) #############


T  = 1000  # Simulation time
Ib = 15.0    # extra current to be injected

net = create_izModularNetwork(0.1)

## Initialise layers
for lr in xrange(len(net.layer)):
  net.layer[lr].v = -65 * np.ones(net.layer[lr].N)
  net.layer[lr].u = net.layer[lr].b * net.layer[lr].v
  net.layer[lr].firings = np.array([])


## SIMULATE
for t in xrange(T):

   # Deliver a constant base current to layer 1

   for i in xrange(len(net.layer)):
       net.layer[i].I = Ib * rand.poisson(0.01, net.layer[i].N)

   net.Update(t)



## Retrieve firings and add Dirac pulses for presentation
firings = np.concatenate([net.layer[i].firings for i in xrange(net.layer)], axis=0)


## Raster plots of firings
if firings.size != 0:
  plt.scatter(firings[:, 0], firings[:, 1] + 1, marker='.')
  plt.xlim(0, T)
  plt.ylabel('Neuron number')
  plt.ylim(0, 1000+1)
  plt.title('Population 1 firings')



plt.show()
