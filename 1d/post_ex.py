from post import *
import multiprocessing as mp
import matplotlib.pyplot as plt

angles =  np.linspace(0,2*math.pi,50)[1:]

X,Y,Z = load_res('res/6/data', 0, 3)
Zf = intep(X,Y,Z)

with mp.Pool(4) as p:
    trajs = p.starmap(
        get_traj,
        [(X,Y,Zf,a0, 0, 25, 2000) for a0 in angles]
    )

plt.pcolor(X,Y,Z)
plt.colorbar()
plot_traj(trajs)
plt.xlabel('Attack angle ($rad$)',fontsize=14)
plt.ylabel('Rotation speed ($rad.s^{-1}$)',fontsize=14)
plt.show()

plt.plot(angles, delta_a(trajs))
plt.xlabel('Attack angle $\\alpha$ ($rad$)',fontsize=14)
plt.ylabel('$\Delta_p\\alpha$',fontsize=14)
plt.grid()
plt.show()

plt.plot(angles, period(trajs))
plt.xlabel('Attack angle ($rad$)',fontsize=14)
plt.ylabel('Period ($s$)',fontsize=14)
plt.grid()
plt.show()
