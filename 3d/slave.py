import loas
import math
import numpy as np
import trimesh
from utils import *

def get_grid(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    thetas = np.arccos(1 - 2*indices/num_pts)
    phis = (math.pi * (1 + 5**0.5) * indices)%(2*math.pi)
    return thetas, phis

Ts, Ps = get_grid(500)

Qs = [v2q(c2v(t,p)) for t,p in np.transpose([Ts,Ps])]
mesh = trimesh.load_mesh("../models/satellite.stl")
mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1])/2) # center the satellite (the mass center should be on 0,0)
drag = loas.RAD(
    sat_mesh = mesh,
    model = loas.models.maxwell(0.10),
    part_per_iteration = 1e5,
    nb_workers = 8
)
drag.start()
res = drag.runSim
    sat_W = loas.Vec(0,0,0),
    sat_Q = Qs,
    sat_speed = 7000,
    sat_temp = 300,
    part_density = 1e-11,
    part_mol_mass = 0.016,
    part_temp = 1800,
    with_drag_coef = False
)
drag.stop()

# save the toque in the satellite frame bc it dont depends on the rotation around (0,0,1) in this frame
Cs = [Qs[index].R2V(torque) for index, (drag, torque) in enumerate(res)]

for name, obj in (('T.npy', Ts), ('P.npy', Ps), ('C.npy', Cs)):
    with open('res_temp/'+name, 'wb') as f:
        np.save(f, obj)
