from random import random
import pandas as pd
import numpy as np

class Particle(object):
    def __init__(self, dimension, xrange, vrange):
        xl, xh = xrange
        vl, vh = vrange
        self.x = np.random.uniform(xl, xh, dimension)
        self.v = np.random.uniform(vl, vh, dimension)
        self.best = self.x
        self.l = [1]*dimension
        self.state = 'Exploration'


class RLMPSO(object):
    def __init__(self, pop_size, dimension, states, actions, xrange, vrange, func):
        self.particles = [Particle(dimension, xrange, vrange) for _ in range(pop_size)]
        self.Q = self.build_q_table(states, actions)
        self.func = func
       # self.particles_state = ['Exploration' * pop_size]
        self.actions = actions
        self.states = states
        self.dimension = dimension
        self.pop_size = pop_size

    def build_q_table(self, states, actions):
        table = pd.DataFrame(
            np.zeros((len(states), len(actions))),  # q_table initial values
            index=states,
            columns=actions  # actions's name
        )
        print(table)    # show table
        return table

    def select_action(self, q_table, state, epsilon=0.9):
        state_actions = q_table.loc[state]
        if random() > epsilon:
            action = np.random.choice(self.actions)
        else:
            action = state_actions.idxmax()
        return action

    def get_env_feedback(self, s, a):
        pass

    #w [0.4 0.9] exploration c1 >= c2 convergence c1 <= c2
    def exploration_convergence(self, i, gBest,  c1=2, c2=1, w=0.4):
        self.particles[i+1].v = w*self.particles[i].v + c1*np.random.uniform(0,1)\
                                *(self.particles[i].best)+c2*np.random.uniform(0,1)\
                                *(gBest-self.particles[i].x)
        self.particles[i+1].x = self.particles[i].x + self.particles[i+1].v

    def jump(self, i, theta, Rmax=600, Rmin=-600):
        self.particles[i].x = self.particles[i].best + np.random.normal(0, theta**2)*(Rmax - Rmin)

    #a is  the  acceleration  factor,  p is  the  descent  parameter  that controls  the  decay  of  the  velocity
    def fine_tunig(self, i, fitcount, MaxFEs, JFEs=20, a=1, p=0.5):
        for d in range(self.dimension):
            j = 1
            while fitcount < MaxFEs and j <= JFEs:
                r = np.random.uniform(-0.5, 0.5)
                self.particles[i].v[d] = a/(j*p)*r + self.particles[i].l[d]
                new_x = self.particles[i].best
                new_x[d] += self.particles[i].v[d]
                if self.func(new_x) < self.func(self.particles[i].best):
                    self.particles[i].best = new_x
                    self.particles[i].l[d] *= 2
                else:
                    self.particles[i].l[d] /= 2
                j += 1

    def update_q_table(self, state, action, q_table, r, alpha=0.2, gama=0.1):
        q_table.loc[state, action] += alpha*(r + gama*max(q_table.loc[action]) - q_table.loc[state, action])

    def rlmpso(self):
        gBest = self.particles[0].best
        for p in self.particles:
            if self.func(p.best) < self.func(gBest):
                gBest = p.best

        fitcount = 0
        Max_FEs = 1000
        while fitcount < Max_FEs:
            for i in range(self.pop_size):
                ox = self.particles[i].x
                action = self.select_action(self.Q, self.particles[i].state)
                if action == 'Exploration':
                    self.exploration_convergence(i, gBest)
                elif action == 'Convergence':
                    self.exploration_convergence(i, gBest, 1, 2)
                elif action == 'High-jump':
                    self.jump(i, 0.9)
                elif action == 'Low-jump':
                    self.jump(i, 0.1)
                elif action == 'Fine-tuning':
                    JEFs = 20
                    self.fine_tunig(i, fitcount, Max_FEs, JEFs)
                    fitcount  += (JEFs-1)

                fitcount += 1
                if self.func(self.particles[i].x) < self.func(ox):
                    r = 1
                else:
                    r = -1

                self.update_q_table(self.particles[i].state, action, self.Q, r)
                self.particles[i].state = action

                if self.func(self.particles[i].x) < self.func(self.particles[i].best):
                    self.particles[i].best = self.particles[i].x

                if self.func(self.particles[i].x) < self.func(gBest):
                    gBest = self.particles[i].x

        return gBest














