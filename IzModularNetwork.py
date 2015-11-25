__author__ = 'js3611'


from IzNetwork import IzNetwork
import numpy as np
import numpy.random as rand

# create a modular network using IzNetwork with
# 8 modules of 100 exhibitory neurons and 1 module of 200 inhibitory neurons

def create_izModularNetwork(p=0, n_ex_layer=8, N_ex=100, N_in=200):
    # p = rewiring probability between exhibitory to exhibitory connections
    # n_ex_layer = number of exhibitory layers
    # N_ex = number of neurons in each exhibitory layer
    # N_in = number of neurons in the inhibitory layer

    # Scaling factors
    F_e2e = 17.0
    F_e2i = 50.0
    F_i2e = 2.0
    F_i2i = 1.0

    # Delay
    D = 1
    Dmax = 20

    idx_in = n_ex_layer  # index of the inhibitory layer
    population = [N_ex] * n_ex_layer + [N_in]  # list of neuron numbers for each layer

    net = IzNetwork(population, Dmax)  # create network of Izhikevich neurons

    # set initial current to zero
    for i in xrange(len(population)):
        net.layer[i].I = np.zeros(population[i])

    # make empty connectivity matrix between each layer
    for i in xrange(len(population)):
        for j in xrange(len(population)):
            net.layer[i].S[j] = np.zeros([population[i], population[j]])

    ######### Izhikevich neuron parameters (with variation) #########

    # exhibitory modules
    for i in xrange(n_ex_layer):
        r = rand.random(N_ex)
        net.layer[i].N = N_ex
        net.layer[i].a = 0.02 * np.ones(N_ex)
        net.layer[i].b = 0.20 * np.ones(N_ex)
        net.layer[i].c = -65 + 15*(r**2)
        net.layer[i].d = 8 - 6*(r**2)

    # inhibitory module
    r = rand.random(N_in)
    net.layer[idx_in].N = N_in
    net.layer[idx_in].a = 0.02 + 0.08*r
    net.layer[idx_in].b = 0.25 - 0.05*r
    net.layer[idx_in].c = -65*np.ones(N_in)
    net.layer[idx_in].d = 2*np.ones(N_in)

    ######### connections and delays #########

    # 1. exhibitory to exhibitory
    # 1000 random intra-connection for each exhibitory layer
    for i in xrange(n_ex_layer):
        S = np.zeros([N_ex, N_ex])
        n_connection = 0
        while n_connection < 1000:
            src = rand.random_integers(0, N_ex-1)
            dst = rand.random_integers(0, N_ex-1)
            if src != dst and S[dst, src] != 1:
                S[dst, src] = 1
                n_connection += 1

        net.layer[i].S[i] = S

    # delay = random integer in [1,20] between exhibitory neurons
    for i in xrange(n_ex_layer):
        for j in xrange(n_ex_layer):
            net.layer[i].factor[j] = F_e2e
            d = np.ceil(20 * rand.random([N_ex, N_ex]))
            net.layer[i].delay[j] = d.astype(int)

    # 2. inhibitory to exhibitory
    # connect to each neuron with random weight in [-1,0]
    # delay = 1
    for i in xrange(n_ex_layer):
        net.layer[i].S[idx_in] = -1 * rand.random([N_ex, N_in])
        net.layer[i].delay[idx_in] = D * np.ones([N_ex, N_in])
        net.layer[i].factor[idx_in] = F_i2e

    # 3. exhibitory to inhibitory
    # each inhibitory neuron has focal connection from 4 exhibitory neuron from the same module
    for i in xrange(N_in):
        # pick a random module
        m = rand.random_integers(0, n_ex_layer-1)
        # pick 4 random neurons
        rand_neurons = rand.random_integers(0, N_ex-1, 4)
        net.layer[idx_in].S[m][i][rand_neurons] = rand.random(4)

    # delay = 1
    for i in xrange(n_ex_layer):
        net.layer[idx_in].factor[i] = F_e2i
        net.layer[idx_in].delay[i] = D * np.ones([N_in, N_ex])

    # 4. connections from inhibitory to inhibitory
    # connect to each neuron with random weight in [-1,0]
    net.layer[idx_in].S[idx_in] = -1 * rand.random([N_in, N_in])
    net.layer[idx_in].factor[idx_in] = F_i2i
    # delay = 1
    net.layer[idx_in].delay[idx_in] = D * np.ones([N_in, N_in])

    ######### rewiring #########

    # for each node in exhibitory layers, with some probability, rewire!
    for l in xrange(n_ex_layer):
        for node in xrange(N_ex):
            S = net.layer[l].S[l]
            edge_indices = [i for i in xrange(N_ex) if S[i, node]]
            # consider each edge
            for edge_idx in edge_indices:
                if rand.random() < p:
                    # remove the edge
                    net.layer[l].S[l][edge_idx, node] = 0
                    # pick a new module
                    rand_module = rand.random_integers(0, n_ex_layer-2)
                    if rand_module >= l:
                        # this makes sure the new module is different from current one
                        rand_module = (rand_module + 1)

                    # pick a new node
                    rand_target = rand.random_integers(0, N_ex-1)
                    while net.layer[rand_module].S[l][rand_target, node]==1:
                        # make sure the new node is not connected already
                        rand_target = rand.random_integers(0, N_ex-1)
                    # rewire
                    net.layer[rand_module].S[l][rand_target, node] = 1

    return net












# 1. consider a random module and pick an edge from n1->n2


# 2. pick another random module & a random neuron n'
# 3. rewire (ie. delete the old edge and create a new edge from n1->n'

