import loas
import vpython as vp
import numpy as np
from numpy import array
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from progress.bar import Bar
import math
import trimesh

Cmat = np.load('res/1/all_rot_mat.npy')
sigma = 2

def smooth_interpolate(Cmat, X, Y, Z, sigma, dim):
    return scipy.interpolate.RegularGridInterpolator(
        (X,Y,Z),
        scipy.ndimage.gaussian_filter(Cmat[:,:,:,dim],sigma)
    )

X = np.linspace(-2*math.pi,2*math.pi,100)
Y = np.linspace(-2*math.pi,2*math.pi,100)
Z = np.linspace(-2*math.pi,2*math.pi,100)

Cxi = smooth_interpolate(Cmat, X, Y, Z, sigma, 0)
Cyi = smooth_interpolate(Cmat, X, Y, Z, sigma, 1)
Czi = smooth_interpolate(Cmat, X, Y, Z, sigma, 2)

def C(Q):
    x,z,y = q2c(Q)
    return loas.utils.tov(
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

def sim_traj(Q0, L0, dt, n):
    I = 1e4*np.eye(3)
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

def plot_all_traj():
    plt.figure().add_subplot(111, projection='3d')
    for z in np.linspace(-2*math.pi,2*math.pi,10):
        for y in np.linspace(-2*math.pi,2*math.pi,10):
            for x in np.linspace(-2*math.pi,2*math.pi,10):
                if x**2+y**2+z**2 > (2*math.pi)**2:
                    continue
                Q0 = c2q(x,y,z)
                traj = [
                    loas.utils.tol(Q.R2V(loas.utils.tov(0,0,1)))
                    for Q in sim_traj(Q0,loas.utils.tov(0,0,0),100,500)
                ]
                normals = []
                for i in range(1,len(traj)-1):
                    a = loas.utils.tov(*traj[i-1])
                    b = loas.utils.tov(*traj[i])
                    c = loas.utils.tov(*traj[i+1])
                    normal = loas.utils.cross(c-b, a-b)
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm>1e-6:
                        normals.append(loas.utils.tol(normal))
                plt.plot(*np.transpose(np.array(normals)))
                #plt.plot(*np.transpose(np.array(traj)))
                print(x,y,z)

def animate_traj(traj):
    mesh = trimesh.load_mesh('../models/ionsat.stl')
    bounds = np.array(mesh.bounds)
    mesh.apply_translation(-(bounds[0] + bounds[1])/2)
    mesh.apply_scale(2)
    satellite = vp.compound([
        vp.triangle(
            vs = [
                vp.vertex(pos=vp.vector(*vertex), normal=vp.vector(*mesh.face_normals[itri]))
                for vertex in triangle
            ]
        )
        for itri, triangle in enumerate(mesh.triangles)
    ])
    axe_x_s = vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(1, 0, 0), shaftwidth=0.01, color=vp.vector(1, 0, 0))
    axe_y_s = vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 1, 0), shaftwidth=0.01, color=vp.vector(0, 1, 0))
    axe_z_s = vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 0, 1), shaftwidth=0.01, color=vp.vector(0, 0, 1))
    satellite = vp.compound([axe_x_s, axe_y_s, axe_z_s, satellite]) # add frame to the satellite
    wind = vp.arrow(pos=vp.vector(0, 0, 3), axis=vp.vector(0, 0, -1), shaftwidth=0.01, color=vp.vector(1, 1, 1))
    prevQ = None
    for Q in traj:
        if prevQ is not None:
            satellite.rotate(
                angle=-prevQ.angle(),
                axis=vp.vector(*prevQ.axis()),
                origin=vp.vector(0,0,0)
            )
        satellite.rotate(
            angle=Q.angle(),
            axis=vp.vector(*Q.axis()),
            origin=vp.vector(0,0,0)
        )
        prevQ = Q
        vp.rate(25)
