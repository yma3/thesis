import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

# GUI class to draw the rotary pendulum. Holds an instance of the rotary pendulum object that we wish to draw and uses
# the pendulum's parameters in pyplot.

# Thetaarm refers to the rotary arm and its corresponding angles
# Alphaarm refers to the pendulum arm and its corresponding angles



class GUI():
    def __init__(self, rotpend):
        self.rotpend = rotpend
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-20, 20])
        self.ax.set_ylim3d([-20, 20])
        self.ax.set_zlim3d([-20, 20])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Rotary Pendulum Simulation')
        self.armTheta, = self.ax.plot([], [], [], color='blue', linewidth=2, antialiased=False)
        self.armAlpha, = self.ax.plot([], [], [], color='red', linewidth=2, antialiased=False)
        self.ball, = self.ax.plot([], [], [], marker='o', color='green', markersize=8, antialiased=False)
        self.center, = self.ax.plot([], [], [], marker='o', color='black', markersize=4, antialiased=False)
        self.scale = 100

    def update(self):
        self.center.set_data([0,0,0], [0,0,0])
        self.center.set_3d_properties([0,0,0])

        thetaarm = self.rotpend.getThetaArmPoints()
        self.armTheta.set_data([thetaarm[0]*self.scale, 0], [thetaarm[1]*self.scale, 0])
        self.armTheta.set_3d_properties([0, 0])

        alphaarm = self.rotpend.getAlphaArmPoints()
        self.armAlpha.set_data([alphaarm[0]*self.scale, thetaarm[0]*self.scale],[alphaarm[1]*self.scale, thetaarm[1]*self.scale])
        self.armAlpha.set_3d_properties([alphaarm[2]*self.scale, 0])

        self.ball.set_data([alphaarm[0]*self.scale], [alphaarm[1]*self.scale])
        self.ball.set_3d_properties([alphaarm[2]*self.scale])

        plt.pause(0.0000000000001)