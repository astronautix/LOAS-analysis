import numpy as np
import scipy.ndimage
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import math

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def load_res(path, axis, sigma = 3):

    a = Path(path)

    Z_temp = []
    for b in a.iterdir():
        with open(b, 'r') as f:
            rot_speed = float(b.name)
            temp = np.array(eval(''.join(f.readlines()).replace('array', 'np.array').replace('Vec', 'np.array')))
            try:
                z = [float(k[1][axis]) for k in temp] #k[0] for drag, k[2] for drag coeff
            except Exception as e:
                print(e, temp)
            Z_temp.append([rot_speed, z])

    Z_temp.sort()

    X = np.linspace(0,2*math.pi,100)
    Y = np.array([i[0] for i in Z_temp])
    Z = np.array([i[1] for i in Z_temp])
    Z = scipy.ndimage.gaussian_filter(Z,sigma) #Z smoothed
    Zf = scipy.interpolate.interp2d(X,Y,Z) #Z function

    X,Y = np.meshgrid(X,Y)
    return X,Y,Z,Zf


class Trajectory:
    def __init__(self, A, DA, dt):
        self.A = A
        self.DA = DA
        self.dt = dt
    def __len__(self):
        return len(self.A)

def get_traj(X, Y, Zf, a0, da0, dt, n):
    def F(X):
        return np.array([
            X[1],
            float(Zf(X[0]%(2*math.pi), X[1]))
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
    return Trajectory(Xs[:,0],Xs[:,1],dt)

@np.vectorize
def plot_traj(traj):
    A, DA = traj.A, traj.DA
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

@np.vectorize
def period(traj):
    A, DA, dt = traj.A, traj.DA, traj.dt
    dai = DA[1]
    for i, da in enumerate(DA):
        if da*dai < 0:
            return 2.*dt*i
    return None

def get_first_zero(A,offset):
    if A[offset] == 0:
        return offset
    for i in range(offset, len(A)-1):
        if A[i]*A[i+1] < 0:
            return i
    return None

@np.vectorize
def delta_e(traj):
    A, DA = traj.A, traj.DA
    i1 = get_first_zero(A,0)
    if i1 is None:
        return i1
    i2 = get_first_zero(A,i1+1)
    if i2 is None:
        return i2
    if DA[i2+1]*DA[i1+1] < 0: # not a period but a semi period
        i2 = get_first_zero(A,i2+1) #we go to the next semi-period
        if i2 is None:
            return None
    if DA[i2+1]*DA[i1+1] < 0: #if did not succeed
        return None

    DE = 1/2*(DA[i2]**2-DA[i1]**2)
    E = 1/2*DA[i1]**2
    return DE/E

@np.vectorize
def delta_a(traj):
    T = period(traj)[()]
    if T is None: return None
    i_demi_T = int(T/traj.dt/2)
    A = traj.A
    M,m = np.argmax(A[:2*i_demi_T]), np.argmin(A[:2*i_demi_T])
    a1 = (A[M]-A[m])/2
    M,m = np.argmax(A[i_demi_T:3*i_demi_T])+i_demi_T, np.argmin(A[i_demi_T:3*i_demi_T])+i_demi_T
    a2 = (A[M]-A[m])/2
    if (a2-a1)/a1 > .1:
        return None
    return 2*(a2-a1)

if __name__ == "__main__":
    import pickle
    angles =  np.linspace(0,math.pi,50)[1:]
    with open('res_perso/crocus_traj_y', 'rb') as f:
        trajs = pickle.load(f)
