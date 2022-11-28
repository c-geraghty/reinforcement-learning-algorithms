
# DYNA-Q+
import numpy as np
from racetrack_env import RacetrackEnv
import random
import time

# YOUR CODE HERE
class ModifiedQAgent():

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

        for i in range(0 ,15):
            for j in range(0 ,19):
                for k in range(-10, 11):
                    for l in range(-10, 11):
                        for action in range(0, 9):


                            self.q[(i , j, k , l), action] = 0


    def policy(self, state):

        qvals = []
        idx = []

        # generate a random number between 0 and 1
        explore_chance = np.random.rand()

        # if less than epsilon - choose random action
        if 1- self.epsilon < explore_chance:

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

    def best_policy(self, state):

        qvals = []
        idx = []

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

    # initialise racetrack enviroment
    env = RacetrackEnv()

    # list of lists of undiscounted returns
    modified_agent_rewards = []

    # loop for 15 trained agents
    for agent in range(0, 15):

        # initialise agent #
        agent = ModifiedQAgent(env)

        # q values for state action pairs
        qvals = agent.q

        # list of undiscounted returns for output - each episode
        undiscounted = []

        # init_state = env.reset()

        # visited dict: SA, S'R -> Times Visited, Timestep Last Seen
        visited = {}

        # model: S,A -> [((S'R) , Probability of this Transition),...]
        model = {}

        # used to track when SA -> S'R was last seen
        timestep = 0

        # loop for 150 episodes
        for i in range(0, 150):

            # terminates episode when true
            terminal = False

            # track undiscounted rewards
            reward = 0

            # initialise S
            s = env.reset()

            while terminal == False:

                # choose action A from state S
                a = agent.policy(s)

                # # new state > S' from A
                new_state = env.step(a)

                s_dash = new_state[0]
                r = new_state[1]

                # add new reward to list of undiscounted rewards
                reward = reward + r

                # if transition hasn't been seen before
                if ((s, a), (s_dash, r)) not in visited:

                    # add to list of transitions
                    # seen 1 time, at timestep
                    visited[((s, a), (s_dash, r))] = (1, timestep)

                # if state/action, state' reward has been seen before
                else:

                    # number of times visited += 1
                    # timestep last seen at = timestep
                    visited[((s, a), (s_dash, r))] = ((visited[((s, a), (s_dash, r))][0] + 1), timestep)

                # choose optimal A' from S'
                a_max = agent.best_policy(s_dash)

                # Q(S, A) = Q(S, A) + alpha[R + discount * Q(S', A') - Q(S, A)]
                qvals[(s, a)] = qvals[(s, a)] + agent.alpha * (
                            r + agent.discount * qvals[(s_dash, a_max)] - qvals[(s, a)])

                # update model
                index = []
                times_visited_list = []
                s_dash_reward_list = []
                counter = 0

                for key in visited:

                    if key[0] == (s, a):

                        # add S'R to list
                        _s_dash = key[1]
                        times_visited = visited[key[0], key[1]][0]

                        counter = counter + times_visited

                        if times_visited_list:

                            if times_visited >= max(times_visited_list) and times_visited_list:
                                s_dash_reward_list.insert(0, _s_dash)
                                times_visited_list.insert(0, times_visited)

                            else:
                                s_dash_reward_list.append(_s_dash)
                                times_visited_list.append(times_visited)

                        else:

                            s_dash_reward_list.append(_s_dash)
                            times_visited_list.append(times_visited)

                probability = [(times_visited_list[i] / counter) for i in range(0, len(times_visited_list))]

                s_r_probabilities = [(s_dash_reward_list[i], probability[i]) for i in range(0, len(probability))]

                model[(s, a)] = s_r_probabilities

                # S <- S' , A <- A'
                s = s_dash

                if new_state[2] == True:
                    terminal = True

                ### INDIRECT RL ###
                ### CHOOSE RANDOM STATE FROM PREVIOUSLY VISITED ###

                for i in range(0, 25):

                    # will return tuple of 2 tuples ((S, A), (S' R))
                    sa_sr = random.choice(list(visited))
                    state_ = sa_sr[0][0]

                    action_timestep = []

                    for key in visited:

                        if key[0][0] == state_:
                            # action taken at S, and timestep
                            action_timestep.append((key[0][1], visited[key][1]))

                    # choose random action and timestep pair
                    idx = np.random.randint(len(action_timestep))
                    chosen_action_time = action_timestep[idx]
                    _action = chosen_action_time[0]
                    _time = chosen_action_time[1]

                    # if only one entry in model for SA
                    # only 1 S'R
                    if len(model[(state_, _action)]) == 1:

                        list_possible_sr = model[(state_, _action)]

                        _state_dash = list_possible_sr[0][0][0]
                        _reward = list_possible_sr[0][0][1]

                    else:

                        prob = np.random.rand()

                        list_possible_sr = model[(state_, _action)]

                        # if random num is less than most likely S'R use that
                        if prob <= list_possible_sr[0][1]:

                            _state_dash = list_possible_sr[0][0][0]
                            _reward = list_possible_sr[0][0][1]

                        # else check if less than sum of most likely and next likely
                        else:

                            prob_pos = list_possible_sr[0][1]

                            for i in range(1, len(list_possible_sr)):

                                prob_pos = prob_pos + list_possible_sr[i][1]

                                if prob <= prob_pos:
                                    _state_dash = list_possible_sr[i][0][0]
                                    _reward = list_possible_sr[i][0][1]

                                    break

                    # best action in a
                    a_max = agent.best_policy(_state_dash)

                    timestep_delta = timestep - _time

                    kn = 1e-5 * np.sqrt(timestep_delta)

                    # update q vals
                    qvals[(state_, _action)] = qvals[(state_, _action)] + agent.alpha * (
                                _reward + + agent.discount * qvals[(_state_dash, a_max)] - qvals[(state_, _action)])

                # increment timestep
                timestep = timestep + 1

            # after episode is finished
            # total reward can be added to list of undiscounted rewards
            undiscounted.append(reward)

        # append list of undiscounted rewards for all episodes for given agent
        modified_agent_rewards.append(undiscounted)





