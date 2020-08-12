try:
    import click
    import trimesh
    import loas
    import math
    import numpy as np
except Exception as e:
    import socket
    print(socket.gethostname(), ':', e)

@click.command()
@click.option('-w')
def run(w):
    mesh = trimesh.load_mesh("../models/crocus/45.stl")
    mesh.apply_scale(.01)
    sat_Q = [loas.Quat(math.cos(angle/2), math.sin(angle/2), 0, 0) for angle in np.linspace(0, 2*math.pi, 100)]
    drag = loas.RAD(
        sat_mesh = mesh,
        model = loas.models.maxwell(0.10),
        part_per_iteration = 1e5,
        nb_workers = 8
    )
    drag.start()
    res = drag.runSim(
        sat_W = loas.Vec(float(w),0,0),
        sat_Q = sat_Q,
        sat_speed = 7000,
        sat_temp = 300,
        part_density = 1e-11,
        part_mol_mass = 0.016,
        part_temp = 1800,
        with_drag_coef = True
    )
    drag.stop()

    with open('res_temp/'+str(w), 'w') as f:
        f.write(str(res))

try:
    run()
except Exception as e:
    import socket
    print(socket.gethostname(), ":", e)
