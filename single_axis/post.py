import numpy as np
from numpy import array
import scipy.ndimage
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import math

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

a = Path('res/3/data')
Z_temp = []
for b in a.iterdir():
    with open(b, 'r') as f:
        rot_speed = float(b.name)
        temp =array(eval(''.join(f.readlines())))
        try:
            z = [float(k[1][0]) for k in temp] #k[0] for drag, k[2] for drag coeff
        except Exception as e:
            print(e, temp)
        Z_temp.append([rot_speed, z])

Z_temp.sort()

X = np.linspace(0,2*math.pi,100)
Y = np.array([i[0] for i in Z_temp])
Z_raw= np.array([i[1] for i in Z_temp])
Z = scipy.ndimage.gaussian_filter(Z_raw,3)
Z_interpolated = scipy.interpolate.interp2d(X,Y,Z)

f = np.zeros(Z_raw.shape)
for i in range(f.shape[0]):
    for j in range(f.shape[1]):
        x = X[j]
        y = Y[i]
        f[i,j] = (Z_interpolated(x, y) - Z_interpolated(x, 0))/y

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

def plot_trajectory(a0, da0, dt, n):
    A, DA = sim_trajectory(a0,da0,dt,n)
    period = 2*math.pi
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
    plt.scatter(A_coll[0][0], DA_coll[0][0], color='blue')
    for i in range(len(A_coll)):
        plt.plot(A_coll[i], DA_coll[i], color='red')

def period_traj(a0, dt, n):
    A, DA = sim_trajectory(a0,0,dt,n)
    dai = DA[1]
    for i, da in enumerate(DA):
        if da*dai < 0:
            return 2*dt*i
    return None

def get_first_zero(A,offset):
    for i in range(offset, len(A)-1):
        if A[i]*A[i+1] < 0:
            return i
    return None

def delta_e(a0,dt,n):
    A, DA = sim_trajectory(a0,0,dt,n)

    i1 = get_first_zero(A,0)
    if i1 is None:
        return i1
    i2 = get_first_zero(A,i1+1)
    if i2 is None:
        return i2
    if DA[i2]*DA[i1] < 0: # not a period but a semi period
        i2 = get_first_zero(A,i2+1) #we go to the next semi-period
        if i2 is None:
            return None
    if DA[i2]*DA[i1] < 0: #if did not succeed
        return None

    DE = 1/2*(DA[i2]**2-DA[i1]**2)
    E = 1/2*DA[i1]**2
    return DE/E
