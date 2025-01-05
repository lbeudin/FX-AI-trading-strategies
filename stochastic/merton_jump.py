import numpy as np
import math
import random

# Project made by Léa Beudin _ merton jump to simulate FX 
class FX_merton:
    def __init__(self, params):
        #paramss = [[1,0,0.004,0.1,0.0003,0.1],[1,0.0001,0.0015,0.05,0.001,0.1],[1,0.0002,0.0015,0.02,0.0005,0.1]]
        self.S, self.mu, self.sigma, self.lamb, self.nu, self.delta = params
        self.last_return = 0
        self.last_positions = []
        self.immediate_return = 0
        self.dt = 0.01 

    def simulate_Merton_Jump(self) :
        dW = math.sqrt(self.dt) * self.gaussian()
        diffusion = (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * dW

        dN = 1 if random.random() < self.lamb * self.dt  else 0
        J = math.exp(self.nu + self.delta * self.gaussian()) if dN == 1 else  1
        self.S = self.S * math.exp(diffusion) * J
        return self.S
            
    def gaussian(self) :
        u = random.random()
        v = random.random()
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)

    def build_series(self, iteration):
        return np.array([self.simulate_Merton_Jump() for i in range(iteration)])


class FX_merton_multiple:
    def __init__(self, merton_list, correl):
        #paramss = [FX_merton([1,0,0.004,0.1,0.0003,0.1]),FX_merton([1,0.0001,0.0015,0.05,0.001,0.1]),FX_merton([1,0.0002,0.0015,0.02,0.0005,0.1]), [1,0.3,0.6]]
        self.merton_list = merton_list
        self.correl = correl
        self.last_return = 0
        self.last_positions = []
        self.immediate_return = 0
        self.dt = 0.01 

    def simulate_Merton_Jump_correl(self) :
        dZ,i = 0
        for merton in merton_list :
            if i == 0 :
                dZ = math.sqrt(self.dt) * merton.gaussian()

            # asset 0 should have correlation of 1
            dW = self.correl[i] * dZ + np.sqrt(1 - self.correl[i] **2) * math.sqrt(self.dt) * merton.gaussian()
            diffusion = (merton.mu - 0.5 * merton.sigma** 2) * self.dt + merton.sigma * dW
            J = math.exp(merton.nu + merton.delta * merton.gaussian()) if random.random() < merton.lamb * self.dt else  1 
            merton.S = merton.S * math.exp(diffusion) * J
            i = i + 1

        return [merton.S for merton in merton_list]
            

    def build_series(self, iteration):
        return np.array([self.simulate_Merton_Jump_correl() for i in range(iteration)])
