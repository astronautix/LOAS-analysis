import loas
import numpy as np
from numpy import array
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from progress.bar import Bar
import math

def build_matrix(src_folder, dest_folder):
    a = Path(src_folder)
    assert a.is_dir()
    assert Path(dest_folder).is_dir()
    assert not Path(dest_folder+'/all_rot_mat.npy').is_file()
    files = list(a.iterdir())
    bar = Bar('Processing', max=len(files))

    C = []
    k = 0
    for file in files:
        bar.next()
        with open(file, 'r') as f:
            x = float(file.name)
            temp = list(eval(''.join(f.readlines())))
            for iy in range(len(temp)):
                for iz in range(len(temp[i])):
                    if temp[iy][iz] is None:
                        temp[iy][iz] = (0,0,0)
                    else:
                        temp[iy][iz] = temp[iy][iz][1][:,0]
            C.append((x,temp))
    C.sort()

    #purge sorting index
    for i in range(len(C)):
        C[i] = C[i][1]
    C = np.array(C)

    bar.finish()
    with open(dest_folder+'/all_rot_mat.npy', 'wb') as f:
        np.save(f, C)
