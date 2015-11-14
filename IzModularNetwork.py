__author__ = 'js3611'


from IzNetwork import IzNetwork
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

# create a modular network using IzNetwork with
# 8 modules of 100 exhibitory neurons and 1 module of 200 inhibitory neurons

F_e2e = 17
F_e2i = 50
F_i2e = 2
F_i2i = 1
D = 1
Dmax = 20
n_ex_layer = 8

N_ex = 100
N_in = 200
idx_in = 8

population = [N_ex] * n_ex_layer + [N_in]
net = IzNetwork(population, Dmax)

# make empty connectivity matrix between each layer
for i in xrange(len(population)):
    for j in xrange(len(population)):
        net.layer[i].S[j] = np.zeros([population[i], population[j]])

for i in xrange(n_ex_layer):
    # make each module (layer) using random connections
    r = rand.random(N_ex)
    net.layer[i].N = N_ex
    net.layer[i].a = 0.02 * np.ones(N_ex)
    net.layer[i].b = 0.20 * np.ones(N_ex)
    net.layer[i].c = -65 + 15*(r**2)
    net.layer[i].d = 8 - 6*(r**2)

    # connectivity
    S = np.zeros([N_ex, N_ex])
    n_connection = 0
    while n_connection < 1000:
        src = rand.random_integers(0, N_ex-1)
        dst = rand.random_integers(0, N_ex-1)
        if src != dst and S[dst, src] != 1:
            S[dst, src] = 1
            n_connection += 1

    net.layer[i].S[i] = S
    net.layer[i].factor[i] = F_e2e
    d = 20 * rand.random(N_ex*N_ex).reshape((N_ex, N_ex))
    net.layer[i].delay[i] = d

# rewire exhibitory neurons

# inhibitory neuron setting

r = rand.random(N_in)
net.layer[idx_in].N = N_in
net.layer[idx_in].a = 0.02 * np.ones(N_in)
net.layer[idx_in].b = 0.25 * np.ones(N_in)
net.layer[idx_in].c = -65 + 15*(r**2)
net.layer[idx_in].d = 2 - 6*(r**2)

# 1. connection from inhibitory to exhibitory with random weight
for i in xrange(n_ex_layer):
    net.layer[i].S[idx_in] = -1 * rand.random(N_ex * N_in).reshape((N_ex, N_in))
    net.layer[i].delay[idx_in] = D * np.ones([N_ex, N_in])
    net.layer[i].factor[idx_in] = F_i2e

# 2. connections from exhibitory to inhibitory
# each inhibitory neuron has focal connection from 4 exhibitory neuron from the same module
for i in xrange(N_in):
    # pick a random module
    m = rand.random_integers(0, n_ex_layer)
    # pick 4 random neurons
    rand_neurons = rand.random_integers(0, N_ex-1, 4)
    S = net.layer[idx_in].S[m]
    S[i][rand_neurons] = rand.random(4)

for i in xrange(n_ex_layer):
    net.layer[idx_in].factor[i] = F_e2i
    net.layer[idx_in].delay[i] = D

# 3. connections from inhibitory to inhibitory
net.layer[idx_in].S[idx_in] = -1 * rand.random(N_in * N_in).reshape((N_in, N_in))
net.layer[idx_in].factor[idx_in] = F_i2i
net.layer[idx_in].delay[idx_in] = D

# Print out the matrix for fun
vertical = {}
for i in xrange(len(population)):
    vertical[i] = np.concatenate(net.layer[i].S.values(), axis=1)
network_matrix = np.concatenate(vertical.values(), axis=0)
plt.matshow(network_matrix, fignum=100, cmap=plt.cm.gray)
plt.show()


# connect each module again randomly

# 1. consider a random module and pick an edge from n1->n2
# 2. pick another random module & a random neuron n'
# 3. rewire (ie. delete the old edge and create a new edge from n1->n'

