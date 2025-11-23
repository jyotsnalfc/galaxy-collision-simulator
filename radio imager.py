#galaxy simulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'

# The following simulation represents two galaxies with random radii and a fixed number of stars colliding and how each would behave gravitationally

#parameters

n_stars = 200 #number of stars per galaxy 
G = 1 #for simulation purposes we use arbitrary units which can be replaced by real world values later
dt = 0.01 #timestep for integration
n = 1500 #number of iterations 
soften = 0.05 #such that stars do not experience infinite gravity at zero distance consistent with real world occurence

#structure of the galaxy
radius = 2 
thickness = 0.2 
m_star = 1
rotational_vel = 1
offset = 10
relative_vel = 0.05

#dark matter halos
mass_halo = 500
radius_halo = 1

def galaxy (N, centre, bulk_vel,
                          radius, rotational_vel, thickness):
    
    cx, cy, cz = centre
    radii = radius * np.sqrt(np.random.rand(N))
    ang = 2 * np.pi * np.random.rand(N)
    
    x = cx + radii * np.cos(ang)
    y = cy + radii * np.sin(ang)
    z = cz + np.random.normal(0, thickness, N)

    vx = -rotational_vel * np.sin(ang) + bulk_vel[0]
    vy = rotational_vel * np.cos(ang) + bulk_vel[1]
    vz = np.random.normal(0, 0.05 * rotational_vel, N) + bulk_vel[2]

    position = np.vstack((x, y, z)).T
    velocity = np.vstack((vx, vy, vz)).T
    M = np.ones(N)
    return position, velocity, M

def acceleration(position, M, G, soften):
    x, y, z = position[:, 0], position[:, 1], position[:,2]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dz = z[:, None] - z[None, :]
    radius2 = dx ** 2 + dy ** 2 + dz ** 2 + soften ** 2
    inverse = radius2 ** -1.5
    np.fill_diagonal(inverse, 0.0)
    ax = -G * (dx * inverse) @ M
    ay = -G * (dy * inverse) @ M
    az = -G * (dz * inverse) @ M
    return np.column_stack((ax, ay, az))

def plummer(position, centre, mass_halo, a, G):
    dx = position - centre
    radius2 = np.sum(dx ** 2, axis = 1) + a**2
    inverser32= radius2 ** -1.5
    return -G * mass_halo * (dx * inverser32[:, None])

def total_acceleration(position, m, halo_centre):

    a = acceleration(position, m, G, soften)
    for c in halo_centre:
        a += plummer(position, c, mass_halo, radius_halo, G)
    return a

centreA = np.array([-offset, 0, 0])
centreB = np.array([offset, 0, 0])
halo_centre = [centreA.copy(), centreB.copy()]

position_A, velocity_A, mass_A = galaxy(
    n_stars, centreA, np.array([+relative_vel, 0, 0]), radius, rotational_vel, thickness
)

position_B, velocity_B, mass_B = galaxy(
    n_stars, centreB, np.array([-relative_vel, 0, 0]), radius, thickness, rotational_vel
)

position = np.vstack((position_A, position_B))
velocities = np.vstack((velocity_A, velocity_B))
mass = np.concatenate((mass_A, mass_B))
labels = np.array([0] * n_stars + [1] * n_stars)

acc = total_acceleration(position, mass, halo_centre)
velocities += 0.5 * acc * dt

plt.ion()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

speeds = np.linalg.norm(velocities, axis=1)
norm_s = (speeds - speeds.min()) / (speeds.ptp() + 1e-6)
colors = cm.plasma(norm_s)

sc = ax.scatter(position[:, 0], position[:, 1], position[:, 2], s=4, c=colors, alpha=0.9, edgecolors = 'none')
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_zlim(-5, 5)
ax.grid(False)                      
ax.xaxis.pane.set_color((0,0,0)) 
ax.yaxis.pane.set_color((0,0,0))
ax.zaxis.pane.set_color((0,0,0))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect([1,1,0.5])  
ax.view_init(elev=20, azim=dt*0.3)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
title = ax.set_title('3D Galaxy Interaction – t = 0.00', color = 'white')
writer = FFMpegWriter(fps=20, bitrate=2000)

with writer.saving(fig, "galaxy.mp4", dpi=100):
    for step in range(n):

        position += velocities * dt

        for gi in (0, 1):
            mask = labels == gi
            if mask.any():
                halo_centre[gi] = position[mask].mean(axis=0)

        acc = total_acceleration(position, mass, halo_centre)
        velocities += acc * dt

        if step % 3 == 0:
            speeds = np.linalg.norm(velocities, axis=1)
            norm_s = (speeds - speeds.min()) / (speeds.ptp() + 1e-6)
            brightness = (speeds / speeds.max()) ** 2
            colors = cm.plasma(brightness)

            sc._offsets3d = (position[:, 0], position[:, 1], position[:, 2])
            sc.set_color(colors)
            title.set_text(f'Galaxy Collision Simulation : t = {step * dt:.2f}')
            
            writer.grab_frame()
            plt.pause(0.001)

plt.ioff()
plt.show()



