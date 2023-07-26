

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
import matplotlib.patches as ptc
from time import time
 
 
def simulate(
    cat_states, cat_controls, t, step_horizon, N, reference, obs, rob_diam, save=False
):

 
    def create_triangle(state=[0, 0, 0], h=1, w=0.5, update=False):
        x, y, th = state
        triangle = np.array([[h, 0], [0, w / 2], [0, -w / 2], [h, 0]]).T
        rotation_matrix = np.array([[cos(th), -sin(th)], [sin(th), cos(th)]])
 
        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]
        
    def create_circle_outline(center, diameter, num_segments = 100, linestyle='dashed', linewidth = 1.5):
        theta = np.linspace(0,2*np.pi, num_segments)
        x = center[0]+ diameter/2 * np.cos(theta)
        y = center[1]+ diameter/2 * np.sin(theta)
        segements = np.array([x,y]).T.reshape(-1,1,2)
        lc = LineCollection(segements, linestyles = linestyle, linewidths = linewidth, color = 'black')
        return lc
    def init():
        return (
            path,
            horizon,
            current_state,
            target_state,
        )
 
    def animate(i):
        # get variables
        x = cat_states[0, 0, i]
        y = cat_states[1, 0, i]
        th = cat_states[2, 0, i]
 
        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)
 
        # update horizon
        x_new = cat_states[0, :, i]
        y_new = cat_states[1, :, i]
        horizon.set_data(x_new, y_new)
 
        # update current_state
        current_state.set_xy(create_triangle([x, y, th], update=True))
 
        # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)

        circle.set_center((x,y))
 
        return (
            path,
            horizon,
            current_state,
            target_state,
            circle
        )
 
    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    min_scale = min(reference[0], reference[1], reference[3], reference[4]) - 2
    max_scale = max(reference[0], reference[1], reference[3], reference[4]) + 2
    ax.set_xlim(left=min_scale, right=max_scale)
    ax.set_ylim(bottom=min_scale, top=max_scale)
 
    # create lines:
    #   path
    (path,) = ax.plot([], [], "k", linewidth=2)
    #   horizon
    (horizon,) = ax.plot([], [], "x-g", alpha=0.5)
    #   current_state
    current_triangle = create_triangle(reference[:3])
    current_state = ax.fill(
        current_triangle[:, 0], current_triangle[:, 1], color="r"
    )
    current_state = current_state[0]
    #   target_state
    target_triangle = create_triangle(reference[3:])
    target_state = ax.fill(
        target_triangle[:, 0], target_triangle[:, 1], color="b"
    )
    target_state = target_state[0]

    circle = ptc.Circle((reference[0],reference[1]), rob_diam/2, edgecolor = 'r', facecolor = 'none', linestyle="--")
    ax.add_patch(circle)


    for ob in obs:
        obstacle = ptc.Circle((ob["x"],ob["y"]),ob["diameter"]/2, edgecolor = 'g', facecolor = 'none', linestyle="--")
        ax.add_patch(obstacle)
 
    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=len(t),
        interval=step_horizon * 100,
        blit=True,
        repeat=True,
    )
    plt.show()
 
    if save == True:
        sim.save("./animation" + str(time()) + ".gif", writer="ffmpeg", fps=30)
 
    return