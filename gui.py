import numpy as np
import math
import matplotlib.pyplot as plt

# GUI class to draw the trebuchet. Holds an instance of the trebuchet object that we wish to draw and uses
# the trebuchet's parameters in pyplot.

class GUI():
    def __init__(self, treb):
        self.treb = treb
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-10, 10), ylim=(-treb.h-0.5, 20))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Trebuchet Simulation')
        self.arm, = self.ax.plot([], [], color='blue', linewidth=2, antialiased=False)
        self.sling, = self.ax.plot([], [], color='red', linewidth=1, antialiased=False)
        self.counterbar, = self.ax.plot([], [], color='blue', linewidth=2, antialiased=False)
        self.counterweight, = self.ax.plot([], [], marker='o', color='blue', markersize=8, antialiased=False)
        self.projectile, = self.ax.plot([], [], marker='o', color='green', markersize=4, antialiased=False)
        self.center, = self.ax.plot(0, 0, marker='o', color='black', markersize=4, antialiased=False)
        self.hbar, = self.ax.plot([0, 0], [0, -self.treb.h], color='blue', linewidth=1, antialiased=False)
        self.ground = self.ax.plot([-1000, 1000], [-self.treb.h, -self.treb.h], color='brown', linewidth=3, antialiased=False)

        self.enterStage3 = False

    def update(self):
        if self.treb.currentStage == 3:
            if not self.enterStage3:
                self.ax.set(xlim=(-10, 300), ylim=(-self.treb.h-1, 120))
                self.enterStage3 = True
            projdata = self.treb.getProj_XY_3()
            self.projectile.set_data(projdata[0], projdata[1])

        else:
            projdata = self.treb.getProjectile()
            armdata = np.array([[self.treb.getWeightArmPoint()[0], self.treb.getArmSlingPoint()[0]],[self.treb.getWeightArmPoint()[1], self.treb.getArmSlingPoint()[1]]])
            slingdata = np.array([[projdata[0], self.treb.getArmSlingPoint()[0]],[[projdata[1], self.treb.getArmSlingPoint()[1]]]])
            cwdata = self.treb.getCounterWeight()
            counterbardata = np.array([[cwdata[0],self.treb.getWeightArmPoint()[0]],[cwdata[1],self.treb.getWeightArmPoint()[1]]])
            self.arm.set_data(armdata[0], armdata[1])
            self.sling.set_data(slingdata[0], slingdata[1])
            self.counterbar.set_data(counterbardata[0], counterbardata[1])
            self.counterweight.set_data(cwdata[0], cwdata[1])
            self.projectile.set_data(projdata[0], projdata[1])




        plt.pause(0.0000000000001)

