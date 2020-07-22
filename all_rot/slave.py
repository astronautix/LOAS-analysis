try:
    import click
    import trimesh
    import numpy as np
    import loas
    import math
except Exception as e:
    import socket
    print(socket.gethostname(), ':', e)

@click.command()
@click.option('-x')
def run(x):
    x = float(x)
    params = locals()
    # load mesh object and resize it
    mesh = trimesh.load_mesh("../models/ionsat.stl")
    bounds = np.array(mesh.bounds)
    mesh.apply_translation(-(bounds[0] + bounds[1])/2) # center the satellite (the mass center should be on 0,0)
    drag = loas.rad.RAD(
        sat_mesh = mesh,
        model = loas.rad.models.maxwell(0.10),
        part_per_iteration = 1e4,
        nb_workers = 8
    )
    drag.start()

    Y = np.linspace(-2*math.pi, 2*math.pi, 10)
    Z = np.linspace(-2*math.pi, 2*math.pi, 10)
    sat_Q = []
    mask = -np.ones((len(Y),len(Z)))

    for iy, y in enumerate(Y):
        for iz, z in enumerate(Z):
            r = math.sqrt(x**2+y**2+z**2)
            if r >= 2*math.pi or r < 1e-6:
                continue
            angle = r
            dir = np.array((x,y,z))
            dir /= np.linalg.norm(dir)
            sat_Q.append(loas.utils.Quaternion(math.cos(angle/2), *(math.sin(angle/2)*dir)))
            mask[iy,iz] = len(sat_Q)-1 #mask[ix,iy] contains the future index of the res that have to go in ix, iy
    res = drag.runSim(
        sat_W = loas.utils.tov(0,0,0),
        sat_Q = sat_Q,
        sat_speed = 7000,
        sat_temp = 300,
        part_density = 1e-11,
        part_mol_mass = 0.016,
        part_temp = 1800,
        with_drag_coef = True
    )
    drag.stop()

    reconstructed_res = []
    for iy in range(len(Y)):
        reconstructed_res.append([])
        for iz in range(len(Z)):
            index = int(mask[iy,iz])
            if index != -1:
                reconstructed_res[-1].append(res[index])
            else:
                reconstructed_res[-1].append(None)

    with open('res_temp/'+str(x), 'w') as f:
        f.write(str(reconstructed_res))

try:
    run()
except Exception as e:
    import socket
    print(socket.gethostname(), ":", e)
