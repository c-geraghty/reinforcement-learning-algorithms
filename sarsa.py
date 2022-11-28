# SARSA
import numpy as np
from racetrack_env import RacetrackEnv
import time


# YOUR CODE HERE
class SARSAAgent():

    def __init__(self, env):

        # when you create new q agent
        # you want to initialise q table for all possible states in grid
        # this is rows x columns

        # alpha value
        self.discount = 0.9
        self.alpha = 0.2
        self.epsilon = 0.15
        self.q = {}

        # initialise q_values arbitrarily
        # initialise returns list

        for i in range(0, 15):
            for j in range(0, 19):
                for k in range(-10, 11):
                    for l in range(-10, 11):
                        for action in range(0, 9):
                            self.q[(i, j, k, l), action] = 0

    def policy(self, state):

        qvals = []
        idx = []

        # generate a random number between 0 and 1
        explore_chance = np.random.rand()

        # if less than epsilon - choose random action
        if 1 - self.epsilon < explore_chance:

            # returns action as int from 0-8 chosen randomly
            return np.random.randint(9)

        else:

            # add all action values for state to list
            qvals = [self.q[(state), action] for action in range(0, 9)]

            max_q = max(qvals)

            # find index of max value(s)
            idx = [i for i in range(0, len(qvals)) if qvals[i] == max_q]

            # if more than one max, choose a random idx value to tiebreak
            if len(idx) > 1:

                random_idx = np.random.randint(len(idx))

                return idx[random_idx]

            # else choose the singular max val
            else:

                return idx[0]


if __name__ == "__main__":

    # SARSA rewards for output
    sarsa_rewards = []

    # initialise racetrack enviroment
    env = RacetrackEnv()

    # loop for 20 trained agents
    for i in range(0, 20):

        # initialise agent
        agent = SARSAAgent(env)

        # q values for state action pairs
        qvals = agent.q

        # list of undiscounted returns for output - each episode
        undiscounted = []

        # loop for 150 episodes
        for ep in range(0, 150):

            # terminates episode when true
            terminal = False

            # initialise undiscounted reward for this episode
            reward = 0

            # initialise S
            s = env.reset()

            while terminal == False:

                # choose action A from state S
                a = agent.policy(s)

                # new state > S' from A
                new_state = env.step(a)

                s_dash = new_state[0]
                r = new_state[1]

                # add new reward to list of undiscounted rewards
                reward = reward + r

                # choose A' from S'
                a_dash = agent.policy(s_dash)

                # Q(S, A) = Q(S, A) + alpha[R + discount * Q(S', A') - Q(S, A)]
                qvals[(s, a)] = qvals[(s, a)] + agent.alpha * (
                            r + agent.discount * qvals[(s_dash, a_dash)] - qvals[(s, a)])

                # S <- S' , A <- A'
                s = s_dash
                a = a_dash

                # if new state is terminal then end
                # only goal state is terminal
                if new_state[2] == True:
                    terminal = True

            # after episode is finished
            # total reward can be added to list of undiscounted rewards
            undiscounted.append(reward)

        # append list of undiscounted rewards for all episodes for given agent
        sarsa_rewards.append(undiscounted)


