
# On-Policy First-Visit Monte-Carlo Control
import time
import numpy as np
from racetrack_env import RacetrackEnv

# YOUR CODE HERE
class MCAgent():

    def __init__(self, env):

        # set parameters
        self.discount = 0.9
        self.epsilon = 0.15

        # init dictionarys for qvals and returns, one for every state action pair
        # q vals set to 0 arbitrarily
        # returns lists set as empty

        self.q = {}
        self.returns = {}

        for i in range(0 ,15):
            for j in range(0 ,19):
                for k in range(-10, 11):
                    for l in range(-10, 11):
                        for action in range(0, 9):


                            self.q[(i , j, k , l), action] = 0
                            self.returns[(i , j, k , l), action] = []




    # epsilon greedy policy
    # choose optimal option with probability 1 - epsilon
    # choose random exploratory option with prob epsilon
    def policy(self, state):


        qvals = []
        idx = []

        # generate a random number between 0 and 1
        explore_chance = np.random.rand()


        # if less than epsilon - choose random action
        if self.epsilon > explore_chance:



            # returns action as int from 0-8 chosen randomly
            return np.random.randint(9)



        # choose best action
        else:

            # add all action values for state to list
            qvals = [self.q[(state), action] for action in range(0 ,9)]

            max_q = max(qvals)

            # find index of max value(s)
            idx = [i for i in range(0, len(qvals)) if qvals[i] == max_q ]

            # if more than one max, choose a random idx value to tiebreak
            if len(idx) > 1:


                random_idx = np.random.randint(len(idx))

                return idx[random_idx]

            # else choose the singular max val
            else:

                return idx[0]



if __name__ == "__main__":


    # initialise mc rewards for output
    mc_rewards = []

    # initialise racetrack env
    env = RacetrackEnv()

    # loop for 20 trained agents
    for i in range (0 ,20):

        # initialise agent
        agent = MCAgent(env)

        qvals = agent.q

        # list for undiscounted returns for each episode
        undiscounted = []

        # loop for 150 episodes
        for ep in range(0 ,150):

            # terminates episode when true
            terminal = False

            # initialise undiscounted reward for this episode
            reward = 0

            # tracks episode of experience - St, At, Rt+1
            experience = []

            # tracks ONLY states in order ST, ST-1.... S0
            visited_states = []

            # create initial state
            s = env.reset()

            while terminal == False:

                # choose action A from state S
                action = agent.policy(s)

                # new state > S' from A
                new_state = env.step(action)

                s_dash = new_state[0]
                r = new_state[1]

                # add to undiscounted return for this episode
                reward = reward + r

                # track old S, A, Rt+1
                experience.insert(0, (s ,action ,r))

                # if new state is terminal then end
                # only goal state is terminal
                if new_state[2] == True:

                    terminal = True

                else:

                    # keep track of state action pair only
                    visited_states.insert(0, s_dash)

                s = s_dash

            # after episode is generated
            # total reward can be added to list of undiscounted rewards
            undiscounted.append(reward)


            ## initialise G =0
            g = 0

            # tracks position in visited for managing list slice
            # necessary for first visit monitoring
            index = 1

            # loop through every time step in experience
            for timestep in experience:

                state_exp = timestep[0]
                action_exp = timestep[1]
                reward_exp = timestep[2]

                # G = gamma*g + R
                g = agent.discoun t *g + reward_exp

                # check if state has been visited already
                # if so, not first visit
                # do nothing
                if state_exp in visited_states[index:]:

                    index = index + 1

                # first visit
                else:

                    index = index + 1

                    # add append g to returns list for S A
                    agent.returns[(state_exp ,action_exp)].append(g)

                    # calculate average of returns list for S A
                    avg = np.mean(agent.returns[(state_exp ,action_exp)])

                    # set q val for S A to average of returns list
                    qvals[(state_exp ,action_exp)] = avg


        # append list of undiscounted rewards for all episodes for given agent
        mc_rewards.append(undiscounted)
