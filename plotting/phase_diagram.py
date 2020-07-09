import numpy as np
from numpy import array
import scipy.ndimage
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import math
a = Path('../res')
Z_temp = []
for b in a.iterdir():
    with open(b, 'r') as f:
        rot_speed = float(b.name)
        temp =array(eval(''.join(f.readlines())))
        z = [float(k[1][0]) for k in temp]
        Z_temp.append([rot_speed, z])

Z_temp.sort()

X = np.linspace(0,2*math.pi,100)
Y = np.array([i[0] for i in Z_temp])
Z = np.array([i[1] for i in Z_temp])
Z = scipy.ndimage.gaussian_filter(Z,5)
Z_interpolated = scipy.interpolate.interp2d(X,Y,Z)
X,Y = np.meshgrid(X,Y)

def sim_trajectory(a0, da0, dt, n):
    def F(X):
        return np.array([
            X[1],
            float(Z_interpolated(X[0]%(2*math.pi), X[1]))
        ])
    Xs = [np.array([a0, da0])]
    for i in range(n):
        X = Xs[-1]
        k1 = F(X)
        k2 = F(X+dt/2*k1)
        k3 = F(X+dt/2*k2)
        k4 = F(X+dt*k3)
        Xs.append(
            Xs[-1]+dt/6*(k1+2*k2+2*k3+k4)
        )
    Xs = np.array(Xs)
    return Xs[:,0],Xs[:,1]

def plot_trajectory(A,DA, period=2*math.pi, color='red'):
    A_coll = [[A[0]]]
    DA_coll = [[DA[0]]]
    for i in range(1, len(A)):
        if A[i-1]//period < A[i]//period:
            A_coll[-1].append(A[i]%period + period)
            DA_coll[-1].append(DA[i])
            A_coll.append([
                A[i-1]%period - period,
                A[i]%period
            ])
            DA_coll.append([
                DA[i-1],
                DA[i]
            ])
        elif A[i-1]//period > A[i]//period:
            A_coll[-1].append(A[i]%period - period)
            DA_coll[-1].append(DA[i])
            A_coll.append([
                A[i-1]%period + period,
                A[i]%period
            ])
            DA_coll.append([
                DA[i-1],
                DA[i]
            ])
        else:
            A_coll[-1].append(A[i]%period)
            DA_coll[-1].append(DA[i])
    for i in range(len(A_coll)):
        plt.plot(A_coll[i], DA_coll[i], color=color)
