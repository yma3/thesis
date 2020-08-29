import math
from scipy import integrate as integrate
import numpy as np
import rotgui as gui
import time

class RotaryPendulum():
    def __init__(self):
        self.Lp = 0.129 # m
        self.Mp = 0.024 # kg
        self.Lr = 0.085 # m
        self.Mr = 0.095 # kg
        self.Jr = 1/3*self.Mr*self.Lr**2
        self.Jp = 1/3*self.Mp*self.Lp**2 # kgm^2
        self.Dr = 0.0015 # Nms/rad
        self.Dp = 0.0005 # Nms/rad
        self.g = 9.8
        self.Tau = 0

        self.alpha_threshold = 0.3
        self.theta_threshold = 2*np.pi
        self.alpha_dot_thresh = 50
        self.theta_dot_thresh = 50


        # STATE = [alpha, theta, alphadot, thetadot]
        self.state = np.array([0.08, 0, 0, 0]) # Start at 0, 0 | 0, 0

        self.ode = integrate.ode(self.eom).set_integrator('vode', nsteps=500, method='bdf')


        self.justfinished = False

    def eom(self):
        alpha = self.state[0]
        theta = self.state[1]
        alphadot = self.state[2]
        thetadot = self.state[3]

        A_11 = -1/2*self.Mp*self.Lp*self.Lr*math.cos(alpha)
        A_12 = self.Mp*self.Lr**2 + 1/4*self.Mp*self.Lp**2 - 1/4*self.Mp*self.Lp**2*math.cos(alpha)**2+self.Jr
        A_21 = self.Jp+1/4*self.Mp*self.Lp**2
        A_22 = -1/2*self.Mp*self.Lp*self.Lr*math.cos(alpha)

        b_1 = self.Tau - self.Dr*thetadot-1/2*self.Mp*self.Lp**2*math.sin(alpha)*math.cos(alpha)*thetadot*alphadot - 1/2*self.Mp*self.Lp*self.Lr*math.sin(alpha)*alphadot**2
        b_2 = -self.Dp*alphadot + 1/4*self.Mp*self.Lp**2*math.cos(alpha)*math.sin(alpha)*thetadot**2 + 1/2*self.Mp*self.Lp*self.g*math.sin(alpha)

        A = np.array([[A_11, A_12], [A_21, A_22]])
        b = np.array([b_1, b_2])



        # A = np.array([[self.Lp*self.Lr*self.Mp*math.cos(alpha)/2, -1*(self.Jr-self.Lp**2*self.Mp*math.cos(2*alpha)/8 + self.Lp**2*self.Mp/8 + self.Lr**2*self.Mp)], \
                     # [self.Jp+self.Lp**2*self.Mp/4, -self.Lp*self.Lr*self.Mp*math.cos(alpha)/2]])
        # b = np.array([self.Lp**2*self.Mp*math.sin(2*alpha)*alphadot*thetadot/4+self.Lp*self.Lr*self.Mp*math.sin(alpha)*alphadot**2/2+self.Dr*thetadot, \
                     # self.Lp**2*self.Mp*math.sin(2*alpha)*thetadot**2/8-self.Lp*self.Mp*self.g*math.sin(alpha)/2+self.Dp*alphadot])
        # print(A.shape, b.shape)
        Ainv = np.linalg.inv(A)
        xmul = np.matmul(Ainv, b)
        # print(xmul)
        qdot = np.array([alphadot, thetadot, xmul[0], xmul[1]])
        # print(qdot)
        return qdot

    def setTau(self, tau):
        self.Tau = tau

    def setState(self, state):
        self.state = state

    def resetPend(self, random=True):
        self.justfinished = None
        self.state = np.zeros(4)
        self.state[0] = 0.08
        if random:
            self.state[0:2] = np.random.uniform(-0.05, 0.05, 1)
            self.state[2:4] = np.random.uniform(-0.05, 0.05, 1)
            # self.state[0] = self.state[0]+np.pi
            # self.state[1:3] = np.random.normal(0, 0.05, 1)
        return self.state



    def update(self, action, dt=0.02):
        self.setTau(action)
        self.ode.set_initial_value(self.state, 0)
        self.state = self.ode.integrate(self.ode.t + dt)
        self.trimStates()

        done = self.state[0] < -self.alpha_threshold \
               or self.state[0] > self.alpha_threshold \
               or self.state[1] < -self.theta_threshold \
               or self.state[1] > self.theta_threshold \
                or abs(self.state[2]) > self.alpha_dot_thresh \
                or abs(self.state[3]) > self.theta_dot_thresh
        done = bool(done)

        if not done:
            # reward = 1-2*abs(self.state[0]) - 2*abs(self.state[1]) - 0.1*abs(self.state[1]) - 0.1*abs(self.state[2])
            reward = 1.0
        elif self.justfinished is None:
            # reward = 1-2*abs(self.state[0]) - 2*abs(self.state[1]) - 0.1*abs(self.state[1]) - 0.1*abs(self.state[2])
            reward = 1.0
            self.justfinished = 0
        else:
            reward = 0

        return np.array(self.state), reward, done, {}




    def updateTest(self):
        self.state[0] = self.state[0] + 0.1
        # self.state[1] = self.state[1] + 0.1
        self.trimStates()

    def trimStates(self):
        self.state[0] = (self.state[0]+np.pi)%(2*np.pi)-np.pi
        # print(self.state[1])
        # self.state[1] = self.state[1]%(2*np.pi)
        # print(self.state[1])

    def getState(self):
        return self.state

    def getThetaArmPoints(self):
        theta = self.state[1]
        armX = self.Lr*math.cos(theta)
        armY = self.Lr*math.sin(theta)
        return armX, armY

    def getAlphaArmPoints(self):
        alpha = self.state[0]
        pendLbase = self.Lp*math.sin(alpha)
        pendZbase = self.Lp*math.cos(alpha)
        X, Y = self.getThetaArmPoints()
        alphaArmX = X+pendLbase*math.sin(self.state[1])
        alphaArmY = Y-pendLbase*math.cos(self.state[1])
        alphaArmZ = pendZbase
        return alphaArmX, alphaArmY, alphaArmZ

    def runSim(self, dt=0.02, totaltime=0.1):
        fail_flag = False
        iterNum = int(totaltime/dt)
        for _ in range(iterNum+1):
            self.update(dt=dt)
            if abs(self.state[0])-np.pi > 0.2:
                break
                fail_flag = True
        return self.state, fail_flag

    def gains(self):
        out = -14.5992*self.state[0] +3.1623*self.state[1] - 0.0658*self.state[2] +1.3422*self.state[3]
        return out


if __name__=="__main__":
    print("Hello World!")
    rotpend = RotaryPendulum()
    # rotpend.eom()
    # rotpend.update()
    print(rotpend.state)
    # print(rotpend.getThetaArmPoints())
    Gui = gui.GUI(rotpend)
    i = 0
    while True:
        actions = rotpend.gains()
        rotpend.update(action=actions)
        if i%2 == 0:
            Gui.update()
        print(rotpend.state)
        i += 1
        # input("WAIT")

        # if rotpend.state[2] < 1e-8 and (rotpend.state[0] - np.pi) < 1e-8:
            # rotpend.resetPend(random=False)
            # time.sleep(2)

    # inputval = input("Press Enter to continue or q to quit: ")
