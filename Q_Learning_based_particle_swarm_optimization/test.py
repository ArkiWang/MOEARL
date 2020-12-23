from cmath import cos, sqrt
from Q_Learning_based_particle_swarm_optimization.QSO import QSO

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

dimension = 10
pop_size = 100
gen_it = 1000
initial_range = (-600, 600)
q0 = 0.0001
qso = QSO(pop_size=pop_size, dimension=dimension,
          func=griewankm, gen_it=gen_it, initial_range=initial_range, q0=q0)
res = qso.qso()
print(res)
print(griewankm(res))