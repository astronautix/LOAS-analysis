import loas
import vpython as vp
import numpy as np
from numpy import array
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pathlib import Path
from progress.bar import Bar
import math
import trimesh
from utils import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sigma = 0
res_folder = '/home/titus/res_temp'

Ps = np.load(res_folder + '/P.npy')
Ts = np.load(res_folder + '/T.npy')
Cmat = np.load(res_folder + '/C.npy')

C_interp = interpolate(Ts, Ps, Cmat, 0.1)

def C(Q):
    t,p = v2c(q2v(Q))
    return Q.V2R(C_interp(t,p))

def W(Q, L, I):
    return Q.V2R(np.linalg.inv(I) @ Q.R2V(L))

def sim_traj(Q0, L0, dt, n):
    I = np.eye(3)
    Q = Q0
    L = L0
    trajectory = []
    for i in range(n):
        L += (C(Q)-3e-5*W(Q,L,I))*dt #calcul du nouveau moment cinétique
        Qnump = Q.vec() + Q.derivative(W(Q,L,I))*dt #calcul de la nouvelle orientation
        Qnump /= np.linalg.norm(Qnump)
        Q = loas.Quat(*Qnump[:,0])
        trajectory.append(Q)
    return trajectory

def plot_all_traj_normals():
    dt = 50
    ax = plt.figure().gca(projection='3d')
    for p in np.linspace(0,2*math.pi,10)[:-1]:
        for t in np.linspace(0,math.pi,10)[:-1]:
            Q0 = v2q(c2v(t,p))
            traj = [q2v(Q) for Q in sim_traj(Q0,loas.Vec(0,0,0),dt,1000)]
            normals = []
            for i in range(1,len(traj)-1):
                normal = (traj[i+1]-traj[i]).cross(traj[i-1]-traj[i])/dt**2
                normals.append(normal.line())
            normals = np.array(normals)
            ax.plot(*np.transpose(np.array(normals)), color='red')
            #plt.plot(*np.transpose(np.array(traj)))
            print(p,t)
    ax.auto_scale_xyz(*[[np.min(normals), np.max(normals)]]*3)
    plt.legend()
    plt.show()

def change_labels(ax):
    ax.set_xticklabels(np.vectorize(int)(np.linspace(0,360,13)[1:-1])) #python3 code to create 90 tick marks
    ax.set_yticklabels(np.vectorize(int)(np.linspace(180,0,13)[1:-1])) #python3 code to create 90 tick marks

def plot_all_traj():
    plt.figure()
    ax = plt.subplot(111, projection='aitoff')
    change_labels(ax)
    for p in np.linspace(0,2*math.pi,4)[:-1]:
        for t in np.linspace(0,math.pi,4)[:-1]:
            Q0 = v2q(c2v(t,p))
            traj = np.array([v2c(q2v(Q)) for Q in sim_traj(Q0,loas.Vec(0,0,0),50,1000)])
            plt.scatter(*v2m(t,p), color='blue')
            plt.plot(*v2m(traj[:,0], traj[:,1]), color='red')
            print(p,t)
    plt.grid()
    plt.show()

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
    satellite = vp.compound([satellite, *[
        vp.arrow(pos=vp.vector(0, 0, 0), axis=axis, shaftwidth=0.01, color=axis)
        for axis in (vp.vector(1,0,0), vp.vector(0,1,0), vp.vector(0,0,1))
    ]]) # add frame to the satellite
    wind = vp.arrow(pos=vp.vector(0, 0, -3), axis=vp.vector(0, 0, 1), shaftwidth=0.01, color=vp.vector(1, 1, 1))
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

def plot2D(pts):
    plt.figure()
    C_sub = np.array([
        [
            C_interp(i,j)
            for i in np.linspace(math.pi, 0, pts)
        ]
        for j in np.linspace(0,2*math.pi,pts)
    ])
    X = np.linspace(-math.pi, math.pi, pts)
    Y =  np.linspace(-math.pi/2, math.pi/2, pts)
    X,Y = np.meshgrid(X,Y)
    for i,name in ((0,'x'),(1,'y'),(2,'z')):
        ax = plt.subplot(311+i, projection='aitoff')
        change_labels(ax)
        plt.pcolor(X,Y,C_sub[:,:,i,0])
        plt.title('$C_{}$'.format(name))
        plt.colorbar()
        plt.grid()
