import vpython as vp
import trimesh
import loas
import numpy as np
import math

def show(mesh, Qs = None):
    """
    if Qs is None, will display the original situation of the satellite
    if Qs is a single Quaternion, will display the satellite in this situation
    if Qs is a list, wil display step by step every situations
    """
    if Qs is None:
        Qs = [loas.Quat(1,0,0,0)]
    elif type(Qs) != list:
        Qs = [Qs]
    bounds = np.array(mesh.bounds)
    mesh.apply_scale(1/np.linalg.norm(mesh.extents)) # auto resize
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
    wind = vp.arrow(pos=vp.vector(0, 0, 3), axis=vp.vector(0, 0, -1), shaftwidth=0.01, color=vp.vector(1, 1, 1))
    prevQ = None
    for Q in Qs:
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
        vp.rate(2)
