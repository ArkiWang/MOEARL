from random import random
import benchmark
import numpy as np
class QSO(object):
    def __init__(self, pop_size, dimension, func, gen_it, initial_range, q0=1):
        self.pop_size = pop_size
        self.dimension = dimension
        self.low, self.high = initial_range
        self.gen_it = gen_it
        self.init_pop(q0)
        # row state column action
        self.Q = np.array([[q0 for _ in range(pop_size)] for _ in range(gen_it+2)])
        self.a = ["imitation", "disturbance"]
        # state of x
        self.s = [i for i in range(pop_size)]
        self.q0 = q0
        self.Q_norm = np.zeros((gen_it+2, pop_size))
        self.func = func
        self.r = [0 for _ in range(pop_size)]
        self.best = []

    def init_pop(self, q0):
        self.x = np.zeros((self.gen_it+2, self.pop_size, self.dimension))
        for k in range(self.gen_it + 1):
            for i in range(self.pop_size):
                for j in range(self.dimension):
                    self.x[k][i][j] = q0


    def normalization_Q(self, k, n):
        sumq = sum(self.Q[n])
        self.Q_norm[n][self.s[k]] = self.Q[n][self.s[k]]/sumq


    #evalue
    def env_feedback(self, i: int, global_best, n, eta=0.2, rp=4, rn=-1) -> float:
        #execute action
        p = random()
        if p >= eta:
            if self.Q[n][self.s[i]] > 0:
                self.x[n+1][i] = self.x[n][i] + eta*(1 - self.Q_norm[n][self.s[i]])\
                          *(global_best - self.x[n][i])
            else:
                self.x[n+1][i], self.Q[n][self.s[i]] = self.q0, self.q0
        else:#
            self.x[n+1][i], self.Q[n][self.s[i]] = np.random.uniform(-600, 600, self.dimension),\
                                                         np.random.uniform(-0.1, 0.1)
            #print(self.x[(i + 1) % self.pop_size])
        r = rp if self.func(self.x[n+1][i]) >= self.func(self.x[n][i]) else rn
        return r

    def update_Q_table(self, i, r, n, alpha=0.001, gama=0.2):
        self.Q[n+1][i] = self.Q[n][self.s[i]] + (self.Q[n][self.s[i]] - self.Q[n-1][self.s[i]])
        self.Q_norm[n+1][i] = self.Q[n][self.s[i]] + alpha*(r + gama *
                        max(self.Q_norm[n])- self.Q[n][self.s[i]])


    def qso(self) ->[]:
        # iteration
        for n in range(1, self.gen_it):
            maxq = self.q0
            maxi = 0
            for i in range(self.pop_size):
                mq = self.Q[n][self.s[i]]
                if mq > maxq:
                    maxq, maxi = mq, i
            global_best = self.x[n][maxi]
            self.best.append(global_best)

            for i in range(self.pop_size):
                self.normalization_Q(i, n)
                r = self.env_feedback(i, global_best, n)
                self.update_Q_table(i, r, n)

            for i in range(self.pop_size):
                mq = self.Q_norm[n+1][self.s[i]]
                if mq > maxq:
                    maxq, maxi = mq, i
            print("{} best individual i:{} f(xi):{} Q[i][a]:{}".format(n, maxi,
                                                    self.func(self.x[n+1][maxi]), self.Q[n+1][maxi]))
        res = global_best
        for b in self.best:
            if self.func(b) < self.func(res):
                res = b

        return res






