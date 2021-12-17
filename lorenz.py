"""
Solving the Lorenz System: 

dxdt = sigma * (y - x)
dydt = x * (rho - z) - y
dzdt = x * y - beta * z 

Brian Howell
UC Berkeley - Mechanical Engineering
bhowell@berkeley.edu
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from matplotlib import animation
from matplotlib import rc
from matplotlib import cm
from scipy import integrate

np.random.seed(1)
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
# plt.rcParams['animation.ffmpeg_path'] = '/usr/local/Homebrew/Cellar/ffmpeg'

###################################################################################################
###################################### Lorenz System ##############################################
###################################################################################################

# define lorenz system
def fLorenz(time, param, sigma, rho, beta):
    output = [sigma * (param[:, 1] - param[:, 0]), 
              param[:, 0] * (rho - param[:, 2]) - param[:, 1], 
              param[:, 0] * param[:, 1] - beta * param[:, 2]]

    return np.asarray(output).T


###################################################################################################
##################################### Numerical Methods ###########################################
###################################################################################################

def fEuler(xSol_):
    # Forward Euler (WORKS GREAT)
    t = 0
    for i in range(0, int(tSim / dt) - 1):
        xSol_[:, i + 1] = xSol_[:, i, :] + dt * fLorenz(t, xSol_[:, i, :], sigma, rho, beta)
    
    return xSol_

def RK4(xSol_):
    # 4th Order Runge Kutta
    
    t = 0
    for i in range(0, int(tSim / dt) - 1):
        t += 1
        # print('i = ', i)
        # compute intermediary values
        k1 = fLorenz(t,          xSol_[:, i, :],               sigma, rho, beta)
        k2 = fLorenz(t + dt / 2, xSol_[:, i, :] + k1 * dt / 2, sigma, rho, beta)
        k3 = fLorenz(t + dt / 2, xSol_[:, i, :] + k2 * dt / 2, sigma, rho, beta)
        k4 = fLorenz(t + dt,     xSol_[:, i, :] + k3 * dt,     sigma, rho, beta)    

        # compute next time step
        xSol_[:, i + 1, :] = xSol_[:, i, :] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return xSol_


###################################################################################################
######################################## Plotting #################################################
###################################################################################################

def plotTrajectory(xSol_):
    # Plot final trajectories
    nTraj = len(xSol_[:, 0, 0])

    fig = plt.figure(figsize=(25, 20))
    ax = plt.axes(projection='3d')
    ax.axis('off')
    index = -1
    for i in range(nTraj):
        ax.plot3D(xSol_[i, :index, 0], xSol_[i, :index, 1], xSol_[i, :index, 2], lw=1, c='black')
        
    ax.scatter(xSol[:, -1, 0], xSol[:, -1, 1], xSol[:, -1, 2], s=500, c='white')
    ax.set_xlabel('X (m)', fontsize=35)
    ax.set_ylabel('Y (m)', fontsize=35)
    ax.set_zlabel('Z (m)', fontsize=35)

    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))
    ax.set_title('Lorenz Sytem: {} particles, {} seconds, RK4'.format(nTraj, tSim), fontsize=25)
    ax.view_init(elev=35., azim=45)

    ax.set_facecolor('xkcd:white')
    # ax.legend(['Particle Position'], fontsize=35)
    ax.view_init(10, 135)

    plt.show


def animate(xSol_):
    """
    The following code animates the trajectory of ALL rays, regardless of qualifications for
    point-cloud reconstruction.

    This requires ffmpeg installed on your computer!!!

    for 1000 time steps, should take about 4-5 minutes
    """

    def update(n):  # Function to create plot
        for line, point, xi in zip(lines, points, xSol_):
            x, y, z = xi[:n].T
            # line.set_data(x, y)
            # line.set_3d_properties(z)

            point.set_data(x[-1:], y[-1:])
            point.set_3d_properties(z[-1:])

        Title.set_text('Solution Animation: n = {0:4f}'.format(n))
        # ax.view_init(30, 0.3 * n)  # uncomment for spinning
        
        fig.canvas.draw()
        return points + lines,  


    # choose a different color for each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, nTraj))

    # Set up animation
    fig = plt.figure(figsize=(25, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')

    ax.set_xlabel('X (m)', fontsize=35)
    ax.set_ylabel('Y (m)', fontsize=35)
    ax.set_zlabel('Z (m)', fontsize=35)
    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))
    ax.view_init(elev=35., azim=45)
    Title = ax.set_title('')

    lines = sum([ax.plot([], [], [], '-', lw=1, c='r') for _ in range(nTraj)], [])
    points = sum([ax.plot([], [], [], 'o', ms=20, c='yellow') for _ in range(nTraj)], [])

    ax.set_facecolor('xkcd:black')
    ax.legend(['Particle Position'])
    ax.view_init(30, 0.3 * 491)
    anim = animation.FuncAnimation(fig, update, frames=len(xSol_[0, :, 0]), interval=50, blit=False)

    rc('animation', html='html5')

    writervideo = animation.FFMpegWriter(fps=30)
    anim.save('anim.mp4', writer=writervideo)


# %%
###################################################################################################
####################################### Simulation ################################################
###################################################################################################

# sytem parameters
sigma = 10.                                                 # system parameters 
rho = 28.                                                   # system parameters
beta = 8. / 3.                                              # system parameters
dt = 0.001                                                  # time step
tSim = 10.                                                  # simulation time (s)

# trajectory parameters
nTraj = 10                                                  # total number of particles
uBound = 15                                                 # upper bound for initial positions
lBound = -15                                                # lower bound for initial positions
X0 = (uBound - lBound) * np.random.rand(nTraj, 3) + lBound  # compute random initial positions

# pre-allocate memory for solution 
xSol = np.zeros((nTraj, int(tSim / dt), 3))                 # dim = (nTraj, timestep, euclidean)
xSol[:, 0, :] = X0                                          # store initial position

if __name__ == "__main__":
    
    # xSol = RK4(xSol)
    xSol = fEuler(xSol)
    
    # plotTrajectory(xSol)
    # animate(xSol)
