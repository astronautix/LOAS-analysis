from post import *
import multiprocessing as mp
import matplotlib.pyplot as plt

angles =  np.linspace(0,2*math.pi,100)[1:-1]

X,Y,Z,Zf = load_res('/home/titus/res_temp', 2)

with mp.Pool(4) as p:
    trajs = p.starmap(
        get_traj,
        [(X,Y,Zf,a0, 0, 2.5, 20000) for a0 in angles]
    )

plt.pcolor(X,Y,Z)
plot_traj(trajs[::10])
plt.xlabel('Attack angle ($rad$)')
plt.ylabel('Rotation speed ($rad.s^{-1}$)')
plt.show()

plt.plot(angles, delta_e(trajs))
plt.xlabel('Attack angle ($rad$)')
plt.ylabel('$\Delta E_m / E_m$')
plt.grid()
plt.show()

plt.plot(angles, delta_e(trajs))
plt.xlabel('Attack angle ($rad$)')
plt.ylabel('Period ($s$)')
plt.grid()
plt.show()
