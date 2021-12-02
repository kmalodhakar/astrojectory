
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
from matplotlib.animation import FuncAnimation

from edads2 import *

main_df = pd.read_csv("nasa.csv")
df = data_preprocess(main_df)
q = 1
given_date = ["2021-12-01", "2022-1-01", "2022-2-01", "2022-3-01", "2022-4-01", "2022-5-01",
             "2022-6-01", "2022-7-01", "2022-8-01", "2022-9-01", "2022-10-01", "2022-11-01", "2022-12-01"]
coords = kepler_solver( a = df['a'][q], e = df['e'][q], 
                        omega = df['asc_node'][q],
                        w =  df['perihelion_arg'][q], 
                        i = df['inclination'][q], 
                        period = df['orbital_period'][q], 
                        given_date = given_date, close_date = df["close_date"][q])

x_array = []
y_array = []
z_array = []
for i in coords:
    x_array.append(i[0])
    y_array.append(i[1])
    z_array.append(i[2])

fig = plt.figure()
ax = plt.axes(projection="3d") 

# axis = plt.axes(xlim =(0, 4),ylim =(-2, 2))
# initializing a line variable
# line, = axis.plot([], [], lw = 3)
cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[150:250:100//len(x_array)]

# data which the line will
# contain (x, y)

def animate(i):
    print(i)
    sctt = ax.scatter3D(x_array[i], y_array[i], z_array[i], c=cmaplist[i], s=50)
    return sctt,

fig = plt.figure()
ax = plt.axes(projection="3d") 

ax.grid(False)
ax.set_facecolor('xkcd:black')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Position of asteroid on a given date (AU)")
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=3, fancybox=True, shadow=True, labelspacing=1)

sctt = ax.scatter3D(0, 0, 0, c='mediumturquoise', s=500)


anim = FuncAnimation(fig, animate, frames = 13, interval = 20, blit = True)


anim.save('/home/kunaala/python_for_ds/astrojectory/test.mp4', writer = 'ffmpeg', fps = 1)






