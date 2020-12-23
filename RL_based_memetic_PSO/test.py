from RL_based_memetic_PSO.RLMPSO import RLMPSO
from cmath import cos, sqrt

def griewankm(xx: []):
    d = len(xx)
    sum = 0
    prod = 1

    for ii in range(d):
        xi = xx[ii]
        sum = sum + (xi-100)**2/4000
        prod = prod * cos((xi-100)/sqrt(ii+1))

    y = sum - prod + 1
    return y

actions = ['Exploration', 'Convergence', 'High-jump', 'Low-jump', 'Fine-tuning']
states = ['Exploration', 'Convergence', 'High-jump', 'Low-jump', 'Fine-tuning']
rlmpso = RLMPSO(pop_size=100,dimension=2, states=states,actions=actions,xrange=(-100, 100),vrange=(-100, 100),func=griewankm)
res = rlmpso.rlmpso()
print(res)
print(griewankm(res))