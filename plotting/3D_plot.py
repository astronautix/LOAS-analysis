import numpy as np
from numpy import array
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import math
a = Path('/home/titus/res')
Z_temp = []
for b in a.iterdir():
    with open(b, 'r') as f:
        rot_speed = float(b.name)
        temp =array(eval(''.join(f.readlines())))
        z = [float(k[1][0]) for k in temp]
        Z_temp.append([rot_speed, z])

Z_temp.sort()

Z = np.array([i[1] for i in Z_temp])
Z = scipy.ndimage.gaussian_filter(Z,5)
Y = [i[0] for i in Z_temp]
X = np.linspace(0,math.pi,100)
X,Y = np.meshgrid(X,Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_xlabel('Attack angle (rad)')
ax.set_ylabel('Rotational speed (rad/s)')
ax.set_zlabel('Intensity of drag torque (N.m)')

plt.show()
