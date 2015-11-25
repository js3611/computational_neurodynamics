__author__ = 'js3611'

from IzModularNetwork import create_izModularNetwork
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

p_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # rewiring probabilities
n_ex_layer=8
N_ex=100
N_in=200
T  = 1000  # Simulation time
Ib = 15.0    # extra current to be injected
window = 50  # mean firing rate averaging window
shift = 20  # mean firing rate shift
step = 10  # bin size in histogram
start_time = 0  # starting time for counting mean firing rate

for p in p_range:
  net = create_izModularNetwork(p)

############# Q1(a) #############

  vertical = {}
  for i in xrange(len(net.layer)):
    vertical[i] = np.concatenate(net.layer[i].S.values(), axis=1)
  network_matrix = np.concatenate(vertical.values(), axis=0)
  plt.matshow(network_matrix, fignum=100, cmap=plt.cm.gray)
  plt.title('Connection matrix (p=%.1f)' %p)
  # plt.show()
  plt.savefig('q1a_%.1f.svg' % p)
  plt.clf()

############# Q1(b) #############

  net = create_izModularNetwork(p, n_ex_layer, N_ex, N_in)

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

    net.layer[n_ex_layer].I = np.zeros(N_in)  # inhibitory layer
    net.Update(t)

  # Retrieve firings data
  firings = net.layer[0].firings
  if len(firings) == 0:
    firings = firings.reshape(0,2)

  num = 0  # accumulative number of neurons
  for i in xrange(1,len(net.layer)):
    num += net.layer[i-1].N
    firings_i = net.layer[i].firings
    if len(firings_i)>0:
      firings_i[:,1] += num  # adjust neuon indices
      firings = np.vstack((firings, firings_i))

  ## Raster plots of firings
  if firings.size != 0:
    plt.scatter(firings[:, 0], firings[:, 1] + 1, marker='.', color='blue')
    plt.xlim(0, T)
    plt.ylabel('Neuron number')
    plt.ylim(0, N_ex*n_ex_layer+N_in+1)
    plt.title('Neuron firings (p=%.1f)' % p)
    plt.savefig('q1b_%.1f.svg' % p)
    plt.clf()

############# Q1(c) #############

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

  for i in xrange(n_ex_layer):
    plt.plot(mfr[:,0], mfr[:,i+1])
  plt.title('Mean firing rate')
  plt.savefig('q1c_%.1f.svg' % p)
  plt.clf()
