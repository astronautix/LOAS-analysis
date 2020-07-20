# 5

Résultats de all_rot compilés en un fichier numpy par `plotting/all_rot.py` et ouvrable par `plotting/all_rot.py`

master.py:
```python
import sys
import os
import numpy as np
import subprocess
import math
import shutil

zs = np.linspace(-2*math.pi, 2*math.pi, 100)
shutil.rmtree('../res_temp', ignore_errors=True)
os.mkdir('../res_temp')
with open('../si.txt', 'r') as si:
    for z in zs:
        target = si.readline().replace('\n','')
        subprocess.Popen([
            'ssh',
            '-oStrictHostKeyChecking=no',
            'titus.senez@{}.polytechnique.fr'.format(target),
            'killall -q python; cd {}; source {}/bin/activate; nohup python slave.py -z {}'.format(os.environ['PWD'], os.environ['VIRTUAL_ENV'], z)
        ], stdout=sys.stdout, stderr=sys.stderr)
        print(target, z)
```

slave.py:
```python
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
@click.option('-z')
def run(z):
    z = float(z)
    params = locals()
    # load mesh object and resize it
    mesh = trimesh.load_mesh("../models/ionsat.stl")
    bounds = np.array(mesh.bounds)
    mesh.apply_translation(-(bounds[0] + bounds[1])/2) # center the satellite (the mass center should be on 0,0)
    drag = loas.rad.RAD(
        sat_mesh = mesh,
        model = loas.rad.models.maxwell(0.10),
        part_per_iteration = 5e4,
        nb_workers = 8
    )
    drag.start()

    X = np.linspace(-2*math.pi, 2*math.pi, 100)
    Y = np.linspace(-2*math.pi, 2*math.pi, 100)
    sat_Q = []
    mask = -np.ones((100,100))

    for ix, x in enumerate(X):
        for iy, y in enumerate(Y):
            r = math.sqrt(x**2+y**2+z**2)
            if r >= 2*math.pi:
                continue
            angle = r
            dir = np.array((x,y,z))
            dir /= np.linalg.norm(dir)
            sat_Q.append(loas.utils.Quaternion(math.cos(angle/2), *(math.sin(angle/2)*dir)))
            mask[ix,iy] = len(sat_Q)-1
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
    for i in range(100):
        reconstructed_res.append([])
        for j in range(100):
            index = int(mask[i,j])
            if index != -1:
                reconstructed_res[-1].append(res[index])
            else:
                reconstructed_res[-1].append(None)

    with open('../res_temp/'+str(z), 'w') as f:
        f.write(str(reconstructed_res))

try:
    run()
except Exception as e:
    import socket
    print(socket.gethostname(), ":", e)
```