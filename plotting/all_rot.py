import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../../LOAS"))
import loas
import numpy as np
from numpy import array
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from progress.bar import Bar
import math

if Path('all_rot_mat.npy').is_file():
    print('Load from saved matrix...')
    C = np.load('all_rot_mat.npy')
else:
    print('Building matrix...')
    a = Path('/home/titus/res_5e4')
    files = list(a.iterdir())
    bar = Bar('Processing', max=len(files))
    C = []
    k = 0
    for b in files:
        bar.next()
        with open(b, 'r') as f:
            z = float(b.name)
            temp = list(eval(''.join(f.readlines())))
            for i in range(len(temp)):
                for j in range(len(temp[i])):
                    if temp[i][j] is None:
                        temp[i][j] = (0,0,0)
                    else:
                        temp[i][j] = temp[i][j][1][:,0]
            C.append((z,temp))
    C.sort()
    for i in range(len(C)):
        C[i] = C[i][1]
    C = np.array(C)
    bar.finish()
    print('Saving matrix...')
    with open('all_rot_mat.npy', 'wb') as f:
        np.save(f, C)

X = np.linspace(-2*math.pi,2*math.pi,100)
Y = np.linspace(-2*math.pi,2*math.pi,100)
Z = np.linspace(-2*math.pi,2*math.pi,100)

def smooth_interpolate(C, X, Y, Z, sigma, dim):
    return scipy.interpolate.RegularGridInterpolator(
        (X,Y,Z),
        scipy.ndimage.gaussian_filter(C[:,:,:,dim],sigma)
    )

sigma = 2
Cxi = smooth_interpolate(C, X, Y, Z, sigma, 0)
Cyi = smooth_interpolate(C, X, Y, Z, sigma, 1)
Czi = smooth_interpolate(C, X, Y, Z, sigma, 2)

Ci = lambda x,y,z: loas.utils.tov(
    float(Cxi((x,y,z))),
    float(Cyi((x,y,z))),
    float(Czi((x,y,z)))
)

def c2q(x,y,z):
    angle = math.sqrt(x**2+y**2+z**2)
    dir = np.array((x,y,z))
    dir /= np.linalg.norm(dir)
    return loas.utils.Quaternion(math.cos(angle/2), *(math.sin(angle/2)*dir))

def q2c(Q):
    angle = Q.angle()
    dir = Q.axis()
    return tuple((angle*dir)[:,0])

def W(Q, L, I):
    return Q.V2R(np.linalg.inv(I) @ Q.R2V(L))

def sim_trajectory(Q0, L0, dt, n):
    I = np.eye(3)
    Q = Q0
    L = L0

    trajectory = []

    for i in range(n):
        x,y,z = q2c(Q)
        L += Ci(x,y,z)*dt #calcul du nouveau moment cinétique
        Qnump = Q.vec() + Q.derivative(W(Q,L,I))*dt #calcul de la nouvelle orientation
        Qnump /= np.linalg.norm(Qnump)
        Q = loas.utils.Quaternion(*Qnump[:,0])
        trajectory.append(Q)

    return trajectory

def plot_all_traj():
    plt.figure().add_subplot(111, projection='3d')
    i = -1
    for z in (0,):#np.linspace(-2*math.pi,2*math.pi,10):
        for x in (0,): # np.linspace(-2*math.pi,2*math.pi,10):
            for y in np.linspace(-2*math.pi,2*math.pi,10):
                i += 1
                if x**2+y**2+z**2 > (2*math.pi)**2:
                    continue
                Q0 = c2q(x,y,z)
                traj = [loas.utils.tol(Q.R2V(loas.utils.tov(0,0,1))) for Q in sim_trajectory(Q0,loas.utils.tov(0,0,0),1000,5000)]
                norm = []
                for i in range(1,len(traj)-1):
                    a = loas.utils.tov(*traj[i-1])
                    b = loas.utils.tov(*traj[i])
                    c = loas.utils.tov(*traj[i+1])
                    n=loas.utils.cross(c-b, a-b)
                    nn=np.linalg.norm(n)
                    if nn>1e-6:
                        norm.append(loas.utils.tol(n/nn))
                plt.plot(*np.transpose(np.array(norm)))
                print(x,y,z)
