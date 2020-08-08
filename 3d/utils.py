import loas
import numpy as np
import math

def v2q(vec): #Q.V2R(loas.Vec(x,z,y)) = (0,0,1)
    assert np.linalg.norm(vec) - 1 < 1e-10
    axis = vec + loas.Vec(0,0,-1)
    if np.linalg.norm(axis) < 1e-6:
        axis = loas.Vec(1,0,0)
    return loas.Quat(0, *axis)

def q2v(Q):
    return Q.R2V(loas.Vec(0,0,-1))

def c2v(theta, phi):
    return loas.Vec(
        math.sin(theta)*math.cos(phi),
        math.sin(theta)*math.sin(phi),
        math.cos(theta)
    )

def v2c(vec):
    assert np.linalg.norm(vec) - 1 < 1e-10
    x,y,z = vec.line()
    p = math.atan2(y,x)%(2*math.pi)
    t = math.atan2(math.sqrt(x**2+y**2), z)
    return t,p

def v2m(t,p):
    return p-math.pi, -(t-math.pi/2)

def distance_on_sphere(t1,p1,t2,p2):
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # https://en.wikipedia.org/wiki/Great-circle_distance
    return math.acos(math.cos(t1)*math.cos(t2)+math.sin(t1)*math.sin(t2)*math.cos(p1-p2))

def interpolate(Ts,Ps,Cs,s):
    assert len(Ts)==len(Ps)==len(Cs)
    nb_pts = len(Ts)
    def C(t,p):
        norm = 0
        sum = 0
        for i in range(nb_pts):
            t2,p2,c = Ts[i], Ps[i], Cs[i]
            dist = distance_on_sphere(t,p,t2,p2)
            if dist < 1e-10:
                return c
            coef = (1/dist)**s
            norm += coef
            sum += coef*c
        return sum/norm
    return C
