import math
from scipy import integrate as integrate
import numpy as np
import utils
import gui
import matplotlib.pyplot as plt
import time

# Trebuchet class that contains the parameters of a trebuchet as well as an implementation of its dynamics
# Equations of motion are taken from VirtualTrebuchet
# Parameters are set to the defaults on VirtualTrebuchet


class Trebuchet():
    def __init__(self):
        # Length in ft, mass in lb, inertia in lb-ft^2
        self.LAl = 6.792 # Length of major arm
        self.LAs = 1.75 # Length opposite to pivot
        self.LAcg = 2.52 # Length to center of grav from pivot
        self.LW = 2 # Length of Weight
        self.LS = 6.833 # Horizontal Length
        self.h = 5 # Height of treb

        self.g = 32.1741 # Grav
        self.mA = 10.65 # Mass of Arm
        self.mW = 98.09 # Mass of Weight # CONTROL PARAMETER
        self.mP = 0.328 # Mass of Projectile # ENVIRO PARAMETER
        self.IA3 = 64.76 # Inertia of Arm
        self.IW3 = 1 # Intertia of Weight
        self.projdia = 0.249 # Diameter of projectile

        self.rho = 0.0765 # Density of Air in lb/ft^3 at 101.3 kPa and 15 degC
        self.Aeff = np.pi*self.projdia**2/4
        self.Cd = 0.483 # Trebuchet 1.0 Constant Cd

        self.referenceAngle = np.pi/4 # CONTROL INPUT / NETWORK OUTPUT: Angle To Fire
        self.WS = 0 # ENV (NETWORK) INPUT: Wind Speed


        # Aq, Wq, Sq, Aw, Ww, Sw
        self.state = np.array([np.pi-math.acos(self.h/self.LAl), np.pi+math.acos(self.h/self.LAl), np.pi-math.asin(self.h/self.LAl), 0, 0, 0])
        self.projstate = np.array([0, 0, 0, 0])
        self.ode1 = integrate.ode(self.eom1).set_integrator('vode', nsteps=500, method='bdf')
        self.ode2 = integrate.ode(self.eom2).set_integrator('vode', nsteps=500, method='bdf')
        self.ode3 = integrate.ode(self.eom3).set_integrator('vode', nsteps=500, method='bdf')

        self.currentStage = 1

        self.final_distance = 0

    def resetStates(self):
        self.currentStage = 1
        self.final_distance = 0
        self.state = np.array([np.pi-math.acos(self.h/self.LAl), np.pi+math.acos(self.h/self.LAl), np.pi-math.asin(self.h/self.LAl), 0, 0, 0])
        self.projstate = np.array([0, 0, 0, 0])


    def getParams(self):
        params = [self.LAl, self.LAs, self.LAcg, self.LW, self.LS, self.h, self.mA, self.mW, self.mP, self.IA3, self.IW3]
        return params

    def setEnviroParam(self, WS):
        # if WS < 0: # wind speed is set to a negative value
            # WS = 0 # set it to 0
        self.WS = WS

    def setControlParam(self, mW, rAng):
        if mW < 0: # if the counterweight is negative,
            self.mW = 0.001 # set it to a very small number
        if rAng < 0: # if the reference angle is negative,
            self.referenceAngle = 0;
        elif rAng > 90: # if the reference angle is more than 90 degrees,
            self.referenceAngle = np.pi/2
        else:
            self.referenceAngle = np.deg2rad(rAng)
            self.mW = mW

    def getInitcond(self):
        return self.state

    def getCounterWeight(self):
        return np.array([self.LAs*math.sin(self.state[0])+self.LW*math.sin(self.state[0]+self.state[1]), \
                         -self.LAs*math.cos(self.state[0])-self.LW*math.cos(self.state[0]+self.state[1])])

    def getWeightArmPoint(self):
        return np.array([self.LAs*math.sin(self.state[0]), -self.LAs*math.cos(self.state[0])])

    def getArmSlingPoint(self):
        return np.array([-self.LAl*math.sin(self.state[0]), self.LAl*math.cos(self.state[0])])

    def getProjectile(self):
        return np.array([-self.LAl*math.sin(self.state[0])-self.LS*math.sin(self.state[0]+self.state[2]),\
                         self.LAl*math.cos(self.state[0])+self.LS*math.cos(self.state[0]+self.state[2])])

    def getAllPoints(self):
        print("CounterWeight: " + str(self.getCounterWeight()))

    def getProjectileAngle(self):
        proj_xdot, proj_ydot = self.getProjectileVelocity()
        projAngle = math.atan(proj_ydot/proj_xdot)
        if proj_xdot <= 0:
            projAngle = np.pi+projAngle
        return projAngle

    def getProjectileVelocity(self):
        proj_xdot = -self.LAl*math.cos(self.state[0])*self.state[3]-self.LS*math.cos(self.state[0]+self.state[2])*(self.state[3]+self.state[5])
        proj_ydot = -self.LAl*math.sin(self.state[0])*self.state[3]-self.LS*math.sin(self.state[0]+self.state[2])*(self.state[3]+self.state[5])
        return np.array([proj_xdot, proj_ydot])

    def getProj_XY_3(self):
        return [self.projstate[0], self.projstate[1]]


    def eom1(self):
        # Aq, Wq, Sq, Aw, Ww, Sw = inputstate
        Aq = self.state[0]
        Wq = self.state[1]
        Sq = self.state[2]
        Aw = self.state[3]
        Ww = self.state[4]
        Sw = self.state[5]
        # Aw, Ww, Sw = devstate
        M_11 = -self.mP*self.LAl**2*(-1+2*math.sin(Aq)*math.cos(Sq)/math.sin(Aq+Sq)) + self.IA3 + self.IW3 + \
               self.mA*self.LAcg**2 + self.mP*self.LAl**2*math.sin(Aq)**2/math.sin(Aq+Sq)**2 + self.mW*(self.LAs**2+self.LW**2+2*self.LAs*self.LW*math.cos(Wq))
        M_12 = self.IW3 + self.LW*self.mW*(self.LW+self.LAs*math.cos(Wq))
        M_21 = M_12
        M_22 = self.IW3 + self.mW*self.LW**2

        r1 = self.g * self.LAcg * self.mA * math.sin(Aq) + self.LAl * self.LS * self.mP * (math.sin(Sq) * (Aw + Sw)**2 + math.cos(Sq) * ( \
                    math.cos(Aq + Sq) * Sw * (Sw + 2 * Aw) / math.sin(Aq + Sq) + (math.cos(Aq + Sq) / math.sin(Aq + Sq) + self.LAl * math.cos(Aq) / (self.LS * math.sin(Aq + Sq))) * Aw ** 2)) \
             + self.LAl * self.mP * math.sin(Aq) * (self.LAl * math.sin(Sq) * Aw ** 2 - self.LS * (math.cos(Aq + Sq) * Sw * (Sw + 2 * Aw) / math.sin(Aq + Sq) + ( \
                    math.cos(Aq + Sq) / math.sin(Aq + Sq) + self.LAl * math.cos(Aq) / (self.LS * math.sin(Aq + Sq))) * Aw ** 2)) / math.sin(Aq + Sq) \
             - self.g * self.mW * (self.LAs * math.sin(Aq) + self.LW * math.sin(Aq + Wq)) - self.LAs * self.LW * self.mW * math.sin(Wq) * (Aw ** 2 - (Aw + Ww) ** 2)

        r2 = -self.LW * self.mW * (self.g * math.sin(Aq + Wq) + self.LAs * math.sin(Wq) * Aw ** 2)


        Aw_dot = (r1*M_22-r2*M_12)/(M_11*M_22-M_12*M_21)
        Ww_dot = -(r1*M_21-r2*M_11)/(M_11*M_22-M_12*M_21)
        Sw_dot = -math.cos(Aq+Sq)*Sw*(Sw+2*Aw)/math.sin(Aq+Sq) - (math.cos(Aq+Sq)/math.sin(Aq+Sq)+self.LAl*math.cos(Aq)/(self.LS*math.sin(Aq+Sq)))*Aw**2 \
                 - (self.LAl*math.sin(Aq)+self.LS*math.sin(Aq+Sq))*Aw_dot/(self.LS*math.sin(Aq+Sq))

        qdot = np.array([Aw, Ww, Sw, Aw_dot, Ww_dot, Sw_dot])
        return qdot

    def eom2(self):
        # Aq, Wq, Sq, Aw, Ww, Sw = inputstate
        Aq = self.state[0]
        Wq = self.state[1]
        Sq = self.state[2]
        Aw = self.state[3]
        Ww = self.state[4]
        Sw = self.state[5]

        M11 = self.IA3 + self.IW3 + self.mA * self.LAcg ** 2 + self.mP * (self.LAl ** 2 + self.LS ** 2 + 2 * self.LAl * self.LS * math.cos(Sq)) + self.mW * ( \
                    self.LAs ** 2 + self.LW ** 2 + 2 * self.LAs * self.LW * math.cos(Wq))
        M12 = self.IW3 + self.LW * self.mW * (self.LW + self.LAs * math.cos(Wq))
        M13 = self.LS * self.mP * (self.LS + self.LAl * math.cos(Sq))
        M21 = M12
        M22 = self.IW3 + self.mW * self.LW ** 2
        M31 = self.LS * self.mP * (self.LS + self.LAl * math.cos(Sq))
        M33 = self.mP * self.LS ** 2

        r1 = self.g * self.LAcg * self.mA * math.sin(Aq) + self.g * self.mP * (self.LAl * math.sin(Aq) + self.LS * math.sin(Aq + Sq)) - self.g * self.mW * ( \
                    self.LAs * math.sin(Aq) + self.LW * math.sin(Aq + Wq)) - self.LAl * self.LS * self.mP * math.sin(Sq) * ( \
                         Aw ** 2 - (Aw + Sw) ** 2) - self.LAs * self.LW * self.mW * math.sin(Wq) * (Aw ** 2 - (Aw + Ww) ** 2)
        r2 = -self.LW * self.mW * (self.g * math.sin(Aq + Wq) + self.LAs * math.sin(Wq) * Aw ** 2)
        r3 = self.LS * self.mP * (self.g * math.sin(Aq + Sq) - self.LAl * math.sin(Sq) * Aw ** 2)

        Aw_dot = -(r1*M22*M33-r2*M12*M33-r3*M13*M22)/(M13*M22*M31-M33*(M11*M22-M12*M21))
        Ww_dot = (r1*M21*M33-r2*(M11*M33-M13*M31)-r3*M13*M21)/(M13*M22*M31-M33*(M11*M22-M12*M21))
        Sw_dot = (r1*M22*M31-r2*M12*M31-r3*(M11*M22-M12*M21))/(M13*M22*M31-M33*(M11*M22-M12*M21))
        qdot = np.array([Aw, Ww, Sw, Aw_dot, Ww_dot, Sw_dot])

        return qdot

    def eom3(self):
        # Px, Py, Pvx, Pvy = inputstate
        Px = self.projstate[0]
        Py = self.projstate[1]
        Pvx = self.projstate[2]
        Pvy = self.projstate[3]

        Pvx_dot = -(self.rho * self.Cd * self.Aeff * (Pvx-self.WS)*math.sqrt(Pvy ** 2+(self.WS-Pvx)**2))/(2*self.mP)
        Pvy_dot = -self.g - (self.rho * self.Cd * self.Aeff * Pvy*math.sqrt(Pvy ** 2+(self.WS-Pvx)**2))/(2*self.mP)

        qdot = np.array([Pvx, Pvy, Pvx_dot, Pvy_dot])
        return qdot




    def update(self, dt):
        prevstate = self.state
        # Stage 1 calculation. The sling is being dragged onto the ground at max tension
        if self.currentStage == 1:
            self.ode1.set_initial_value(self.state, 0)
            self.state = self.ode1.integrate(self.ode1.t + dt)
            acc = (utils.slingvel(self.LAl, self.LAs, self.state[0], self.state[1], self.state[3], self.state[5]) - \
                  utils.slingvel(self.LAl, self.LAs, prevstate[0], prevstate[1], prevstate[3], prevstate[5]))/dt
            theta = self.state[0] + self.state[2] - np.pi
            tension = -acc[1]*self.mP/math.sin(theta)
            diff = self.mP*self.g - tension
            if(diff<0):
                self.currentStage = 2

        # Stage 2 calculation. The sling is off of the ground and swinging in an arc
        elif self.currentStage == 2:
            self.ode2.set_initial_value(self.state, 0)
            self.state = self.ode2.integrate(self.ode2.t + dt)
            if (self.referenceAngle-self.getProjectileAngle()) > 0:
                self.currentStage = 3
                projp = self.getProjectile()
                projv = self.getProjectileVelocity()
                self.projstate = np.array([projp[0], projp[1], projv[0], projv[1]])

        # Stage 3 calculation. The projectile enters into freefall until it reaches the ground.
        elif self.currentStage == 3:
            self.ode3.set_initial_value(self.projstate, 0)
            self.projstate = self.ode3.integrate(self.ode3.t + dt)
            if self.projstate[1] < -self.h:
                self.final_distance = self.projstate[0]
                return False

        for i in range(3):
            self.state[i] = (self.state[i]) % (2*np.pi)
        return True

    def runsim(self, dt=0.01):
        while(self.update(dt=dt)):
            continue
        return self.final_distance


if __name__ == "__main__":
    distances = []
    masses = []
    print("Hello World!")
    Trebuchet = Trebuchet()
    Trebuchet.setEnviroParam(0)
    # Trebuchet.setControlParam(Trebuchet.mW, 30) # reference angles are set in degrees
    Trebuchet.setControlParam(Trebuchet.mW, 45) # reference angles are set in degrees
    for i in range(50):
        Trebuchet.setControlParam(100, 30+i*1)
        distances.append(Trebuchet.runsim())
        masses.append(30+i*1)
        Trebuchet.resetStates()

    plt.plot(masses, distances)
    plt.show()
    Gui_obj = gui.GUI(Trebuchet)
    Gui_obj.update()
    t = np.linspace(0,1,51)
    i = 0
    while(Trebuchet.update(dt=0.01)):
        if i%5 == 0:
            Gui_obj.update()
            print("STEP: " + str(i) + " || PROJ: " + str(Trebuchet.projstate))
    time.sleep(3)
