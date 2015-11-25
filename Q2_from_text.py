import numpy as np
import matplotlib.pyplot as plt

data = np.array([]).reshape(0,2)
with open('mi.txt') as f:
    for raw_line in f.readlines():
        line = raw_line.strip().replace(" ", "")
        integral, rewiring_p = line.split(',')
        data = np.vstack((data,[float(integral), float(rewiring_p)]))
        
plt.scatter(data[:,0], data[:,1],marker='.')
plt.xlim((0,1))
plt.ylim((0,5))
plt.title("Integration I(S)")
plt.xlabel("Rewiring probability p")
plt.ylabel("Integration (bits)")
plt.plot()
