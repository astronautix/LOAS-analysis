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

sigma = 0
res_folder = '/home/titus/res_temp'

Ps = np.load(res_folder + '/P.npy')
Ts = np.load(res_folder + '/T.npy')
Cmat = np.load(res_folder + '/C.npy')

Cx = interpolate(Ts, Ps, Cmat[:,0,0], 0.1)
Cy = interpolate(Ts, Ps, Cmat[:,1,0], 0.1)
Cz = interpolate(Ts, Ps, Cmat[:,2,0], 0.1)

def C(Q):
    t,p = v2c(q2v(Q))
    C_sat = loas.utils.tov(Cx(t,p), Cy(t,p), Cz(t,p))
    return Q.V2R(C_sat)

def W(Q, L, I):
    return Q.V2R(np.linalg.inv(I) @ Q.R2V(L))

def sim_traj(Q0, L0, dt, n):
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

def plot_all_traj():
    plt.figure().add_subplot(111, projection='3d')
    for z in (0,): #np.linspace(-2*math.pi,2*math.pi,10):
        for y in np.linspace(-math.pi,math.pi,10):
            for x in np.linspace(-math.pi,math.pi,10):
                if x**2+y**2+z**2 > (math.pi)**2:
                    continue
                Q0 = c2q(x,y,z)
                traj = [
                    loas.utils.tol(Q.R2V(loas.utils.tov(0,0,1)))
                    for Q in sim_traj(Q0,loas.utils.tov(0,0,0),2000,1000)
                ]
                normals = []
                for i in range(1,len(traj)-1):
                    a = loas.utils.tov(*traj[i-1])
                    b = loas.utils.tov(*traj[i])
                    c = loas.utils.tov(*traj[i+1])
                    normal = loas.utils.cross(c-b, a-b)
                    normals.append(loas.utils.tol(normal))
                plt.plot(*np.transpose(np.array(normals)))
                #plt.plot(*np.transpose(np.array(traj)))
                print(x,y,z)

def animate_traj(traj):
    mesh = trimesh.load_mesh('../models/satellite.stl')
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


def plot(C):
    fig = plt.figure()
    ax = fig.add_subplot( 1, 1, 1, projection='3d')

    PP = np.linspace( 0, 2 * np.pi, 120)
    TT = np.linspace( 0, np.pi, 60 )

    # create the sphere surface
    XX = np.outer(np.cos(PP), np.sin(TT))
    YY = np.outer( np.sin( PP ), np.sin( TT ) )
    ZZ = np.outer( np.ones( np.size( PP ) ), np.cos( TT ) )

    WW = XX.copy()
    for i,p in enumerate(PP):
        for j,t in enumerate(TT):
            WW[i,j] = C(t,p)
    WW -= np.amin(WW)
    WW = WW / np.amax( WW )
    myheatmap = WW
    ax.plot_surface( XX, YY,  ZZ, cstride=1, rstride=1, facecolors=cm.jet( myheatmap ) )
    plt.show()
