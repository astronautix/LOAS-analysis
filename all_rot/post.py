import loas
import numpy as np
from numpy import array
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from progress.bar import Bar
import math

def build_matrix(src_folder, dest_folder):
    a = Path(src_folder)
    assert a.is_dir()
    assert Path(dest_folder).is_dir()
    assert not Path(dest_folder+'/all_rot_mat.npy').is_file()
    files = list(a.iterdir())
    bar = Bar('Processing', max=len(files))

    C = []
    k = 0
    for file in files:
        bar.next()
        with open(file, 'r') as f:
            z = float(file.name)
            temp = list(eval(''.join(f.readlines())))
            for i in range(len(temp)):
                for j in range(len(temp[i])):
                    if temp[i][j] is None:
                        temp[i][j] = (0,0,0)
                    else:
                        temp[i][j] = temp[i][j][1][:,0]
            C.append((z,temp))
    C.sort()

    #purge sorting index
    for i in range(len(C)):
        C[i] = C[i][1]
    C = np.array(C)

    bar.finish()
    with open(dest_folder+'/all_rot_mat.npy', 'wb') as f:
        np.save(f, C)


def get_C(src):
    """
    Returns the interpolating function that gives the
    """
    assert Path(src).is_file

    def smooth_interpolate(C, X, Y, Z, sigma, dim):
        return scipy.interpolate.RegularGridInterpolator(
            (X,Y,Z),
            scipy.ndimage.gaussian_filter(C[:,:,:,dim],sigma)
        )

    X = np.linspace(-2*math.pi,2*math.pi,100)
    Y = np.linspace(-2*math.pi,2*math.pi,100)
    Z = np.linspace(-2*math.pi,2*math.pi,100)

    sigma = 2
    Cxi = smooth_interpolate(C, X, Y, Z, sigma, 0)
    Cyi = smooth_interpolate(C, X, Y, Z, sigma, 1)
    Czi = smooth_interpolate(C, X, Y, Z, sigma, 2)

    def C(Q):
        x,z,y = q2c(Q)
        return loas.utils.tov(
            float(Cxi((x,y,z))),
            float(Cyi((x,y,z))),
            float(Czi((x,y,z)))
        )

    return C

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

def sim_trajectory(C, Q0, L0, dt, n):
    I = np.eye(3)
    Q = Q0
    L = L0
    trajectory = []
    for i in range(n):
        L += C(Q)*dt #calcul du nouveau moment cinétique
        Qnump = Q.vec() + Q.derivative(W(Q,L,I))*dt #calcul de la nouvelle orientation
        Qnump /= np.linalg.norm(Qnump)
        Q = loas.utils.Quaternion(*Qnump[:,0])
        trajectory.append(Q)
    return trajectory

def plot_all_traj(C):
    plt.figure().add_subplot(111, projection='3d')
    for z in (0,):#np.linspace(-2*math.pi,2*math.pi,10):
        for x in (0,): # np.linspace(-2*math.pi,2*math.pi,10):
            for y in np.linspace(-2*math.pi,2*math.pi,10):
                if x**2+y**2+z**2 > (2*math.pi)**2:
                    continue
                Q0 = c2q(x,y,z)
                traj = [
                    loas.utils.tol(Q.R2V(loas.utils.tov(0,0,1)))
                    for Q in sim_trajectory(C,Q0,loas.utils.tov(0,0,0),1000,5000)
                ]
                normals = []
                for i in range(1,len(traj)-1):
                    a = loas.utils.tov(*traj[i-1])
                    b = loas.utils.tov(*traj[i])
                    c = loas.utils.tov(*traj[i+1])
                    normal = loas.utils.cross(c-b, a-b)
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm>1e-6:
                        normals.append(loas.utils.tol(normal/normal_norm))
                plt.plot(*np.transpose(np.array(normals)))
                print(x,y,z)
