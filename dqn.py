import random
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import rotarypend
import rotgui as gui
import matplotlib.pyplot as plt
import sys
import scipy.io as sio

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.998

# Deep Q-Network implementation. Adapted a format similar to OpenAI Gym environments.
# General implementation uses a Rotary pendulum class as an environment, with predetermined observation and action states
# Running the rotarypendulum() function trains the algorithm for a set number of epochs and outputs the loss as a graph
    # It also shows an example of the pendulum in action after training
    # The trained model is then saved in a specified directory
# Load model loads a pretrained model and shows its performance at a random initialized state


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Define model architecture here
        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    # Save model to a file
    def save(self, fileloc):
        self.model.save(fileloc)

    # Load model from a file
    def load(self, fileloc):
        self.model = load_model(fileloc)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    # Learn from memory
    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

# Discretized action space for Rotary Pendulum based on TORQUE
def discretizeActions(action):
    # torque = (action-3)*0.04/3 # For 7
    actions = {0:-0.02, 1:-0.015, 2:-0.01, 3:-0.005, 4:0, 5:0.005, 6:0.01, 7:0.015 , 8:0.02 }
    # actions = {0:-0.015, 1:-0.01, 2:-0.005, 3:0, 4:0.005, 5:0.01, 6:0.015}
    # actions = {0:-0.015, 1:-0.01, 2:-0.005, 3:0.005, 4:0.01, 5:0.015}
    torque = actions[action]
    return torque

# Test
def testRun(env, dqn, gui):
    state = env.resetPend()
    dqn_solver = dqn
    pendgui = gui
    done = 0
    state = np.reshape(state, [1, 4])
    for t in range(200):
        # action = dqn_solver.act(state)
        q_values = dqn_solver.model.predict(state)
        action = np.argmax(q_values[0])
        torque = discretizeActions(action)
        # if done: torque = 0
        state_next, reward, terminal, info = env.update(torque)
        done = terminal
        state_next = np.reshape(state_next, [1, 4])
        state = state_next
        if t%1 == 0:
            pendgui.update()

# Testrun and save as Matlab matrices
def saveRun(env, dqn, gui):
    states = []
    time = []
    state = env.resetPend()
    print(state)
    state[1:4] = 0
    print(state)
    dqn_solver = dqn
    pendgui = gui
    done = 0
    state = np.reshape(state, [1, 4])
    for t in range(201):
        states.append(state)
        time.append(t*0.02)
        # action = dqn_solver.act(state)
        q_values = dqn_solver.model.predict(state)
        action = np.argmax(q_values[0])
        torque = discretizeActions(action)
        print(action)
        # if done: torque = 0
        state_next, reward, terminal, info = env.update(torque)
        done = terminal
        state_next = np.reshape(state_next, [1, 4])
        state = state_next
        if t%1 == 0:
            pendgui.update()

    states = np.array(states)
    time = np.array(time)
    sio.savemat('statearr_rotpend2.mat', {'states': states})
    sio.savemat('timearr_rotpend2.mat', {'time': time})

EP_LEN = 500

def rotarypendulum():
    last_deq = deque(maxlen=20)
    # env = gym.make(ENV_NAME)
    env = rotarypend.RotaryPendulum()
    pendgui = gui.GUI(env)
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = 4
    action_space = 6
    dqn_solver = DQNSolver(observation_space, action_space)
    ep_r = []
    ep = []
    lastbest = 0
    done_done = 0
    for run in range(5000):
        # run += 1
        state = env.resetPend()
        state = np.reshape(state, [1, observation_space])
        step = 0
        foundbest = 0
        for t in range(EP_LEN):
            sys.stdout.write('\r' + str(t) + ' | ')
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            torque = discretizeActions(action)
            state_next, reward, terminal, info = env.update(torque)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal or t == EP_LEN-1:
                if np.mean(last_deq) > 390:
                    done_done = 1
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step) + ", avg: " + str(np.mean(last_deq)))
                if step > lastbest:
                    lastbest = step
                    foundbest = 1
                ep.append(run)
                last_deq.append(step)
                ep_r.append(np.mean(last_deq))
                # score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()
        if done_done:
            input("Waiting on confirmation...")
            dqn_solver.save('dqn9_3.h5')
            testRun(env, dqn_solver, pendgui)
            break
        # if run % 20 == 0 or foundbest:
            # testRun(env, dqn_solver, pendgui)
        # if run % 5 == 0:
            # plt.plot(ep, ep_r, 'k')
            # plt.pause(0.0000000000001)
    plt.figure(2)
    plt.title('Training Reward v Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.plot(ep, ep_r)
    plt.show()

def loadtest():
    env = rotarypend.RotaryPendulum()
    pendgui = gui.GUI(env)
    observation_space = 4
    action_space = 9
    dqn_solver = DQNSolver(observation_space, action_space)
    dqn_solver.load('dqn9_2.h5')
    input("Waiting on confirmation...")
    saveRun(env, dqn_solver, pendgui)


if __name__ == "__main__":
    # rotarypendulum()
    loadtest()