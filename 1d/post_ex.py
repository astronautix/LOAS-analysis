from post import *
import multiprocessing as mp
import matplotlib.pyplot as plt

angles =  np.linspace(0,math.pi,50)[1:]

X,Y,Z,Zf = load_res('res/5/data', 0, 3)

with mp.Pool(4) as p:
    trajs = p.starmap(
        get_traj,
        [(X,Y,Zf,a0, 0, 0.025, 800000) for a0 in angles]
    )

plt.pcolor(X,Y,Z)
plot_traj(trajs)
plt.xlabel('Attack angle ($rad$)',fontsize=14)
plt.ylabel('Rotation speed ($rad.s^{-1}$)',fontsize=14)
plt.show()

plt.plot(angles, delta_a0(trajs))
plt.xlabel('Attack angle ($rad$)',fontsize=14)
plt.ylabel('$\Delta E_m / E_m$',fontsize=14)
plt.grid()
plt.show()

plt.plot(angles, period(trajs))
plt.xlabel('Attack angle ($rad$)',fontsize=14)
plt.ylabel('Period ($s$)',fontsize=14)
plt.grid()
plt.show()
