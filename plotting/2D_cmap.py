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
Y = np.array([i[0] for i in Z_temp])
X = np.linspace(0,math.pi,100)
X,Y = np.meshgrid(X,Y)


dza = []
dzda = []
iza = []

for i in range(1, Z.shape[0]-1):
    dza.append([])
    dzda.append([])
    iza.append([0])
    for j in range(1, Z.shape[1]-1):
        dza[-1].append(
            (Z[i,j+1]-Z[i,j-1])/(X[i,j+1]-X[i,j-1])
        )
        dzda[-1].append(
            (Z[i+1,j]-Z[i-1,j])/(Y[i+1,j]-Y[i-1,j])
        )
        iza[-1].append(iza[-1][-1] + (X[i,j+1]-X[i,j-1])*Z[i,j])

dza = np.array(dza)
dzda = np.array(dzda)

w0 = np.sqrt(-dza)
Q = -w0/dzda

plt.pcolor(X, Y, Z)
plt.xlabel('Attack angle (rad)')
plt.ylabel('Rotational speed (rad/s)')
plt.colorbar()
plt.show()
