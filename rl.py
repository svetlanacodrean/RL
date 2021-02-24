import random
import matplotlib.pyplot as plt
import numpy as np


# 0 - empty, 1 - wall, 2 - snake pit, 3 - reward
class Gridworld:
    def __init__(self, dimensions, coords_walls, coords_snake_pits, coords_rewards, rewards, actions, policy):
        self.dimensions = dimensions
        self.grid = np.zeros(shape=dimensions)
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                if (i,j) in coords_walls:
                    self.grid[i][j] = 1
                elif (i,j) in coords_snake_pits:
                    self.grid[i][j] = 2
                elif (i,j) in coords_rewards:
                    self.grid[i][j] = 3
        print(self.grid)
        # plt.imshow(self.grid, cmap='tab20', interpolation='nearest')
        # plt.show()
        self.grid_rewards = np.zeros(shape=dimensions)
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                if self.grid[i][j] == list(rewards.keys())[0]:
                    self.grid_rewards[i][j] = list(rewards.values())[0]
                elif self.grid[i][j] == list(rewards.keys())[1]:
                    self.grid_rewards[i][j] = list(rewards.values())[1]
                elif self.grid[i][j] == list(rewards.keys())[2]:
                    self.grid_rewards[i][j] = list(rewards.values())[2]
                elif self.grid[i][j] == list(rewards.keys())[3]:
                    self.grid_rewards[i][j] = list(rewards.values())[3]
        # print(self.grid_rewards)
        self.actions = actions
        self.policy = policy
        self.reward_map = np.zeros(shape=dimensions, dtype=[('x', 'i4'), ('y', 'i4')])
        # print(self.reward_map)

    def is_wall_or_border(self, coords_tuple):
        if coords_tuple[0] < 0 or coords_tuple[0] >= dimensions[0]:
            return True
        if coords_tuple[1] < 0 or coords_tuple[1] >= dimensions[1]:
            return True
        if coords_tuple in coords_walls:
            return True
        return False

    def is_over(self, coords_tuple):
        if coords_tuple in coords_snake_pits or coords_tuple in coords_rewards:
            return True
        return False

    def get_random_location(self):
        x = random.choice(range(dimensions[0]))
        y = random.choice(range(dimensions[1]))
        state = (x, y)
        while self.is_wall_or_border(state) or self.is_over(state):
            x = random.choice(range(9))
            y = random.choice(range(9))
            state = (x, y)
        return state

    def make_move(self, state_tuple, action):
        move = (state_tuple[0] + action[0], state_tuple[1] + action[1])
        if self.is_wall_or_border(move):
            next_state_tuple = state_tuple
            reward = -1
        else:
            next_state_tuple = move
            reward = self.grid_rewards[next_state_tuple[0]][next_state_tuple[1]]
        return reward, next_state_tuple

    def get_value_for_state(self, state_tuple):
        value = 0
        action = random.choice(self.actions)
        move = (state_tuple[0] + action[0], state_tuple[1] + action[1])
        if self.is_wall_or_border(move):
            next_state_tuple = state_tuple
            reward = -1
            # print("wall", next_state_tuple)
        elif self.is_over(move):
            # print("finish", move)
            return self.grid_rewards[move[0]][move[1]]
        else:
            next_state_tuple = move
            reward = self.grid_rewards[next_state_tuple[0]][next_state_tuple[1]]
            # print("nothing", next_state_tuple)
        next_state_value = self.get_value_for_state(next_state_tuple)
        value += reward + next_state_value
        # put the value in the agent rewards map
        self.reward_map[state_tuple[0]][state_tuple[1]][0] += value
        self.reward_map[state_tuple[0]][state_tuple[1]][1] += 1
        # print(value)
        return value


class Q_Agent():
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=1):
        self.environment = environment
        self.q_table = dict()
        for x in range(environment.dimensions[0]):
            for y in range(environment.dimensions[1]):
                self.q_table[(x, y)] = {(-1,0): 0, (0,1): 0, (1,0): 0, (0,-1): 0}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, available_actions, state_tuple):
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.choice(available_actions)
        else:
            q_values_of_state = self.q_table[state_tuple]
            maxValue = max(q_values_of_state.values())
            max_actions = [k for k, v in q_values_of_state.items() if v == maxValue]
            action = random.choice(max_actions)
        return action

    def learn(self, old_state, reward, new_state, action):
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]

        self.q_table[old_state][action] = (1 - self.alpha) * current_q_value + self.alpha * (
                    reward + self.gamma * max_q_value_in_new_state)


def playQ(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    reward_per_episode = []
    won = 0
    lost = 0

    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        # while step < max_steps_per_episode and game_over != True:  # Run until max steps or until game is finished
        #     old_state = environment.get_random_location()
        #     action = agent.choose_action(environment.actions, old_state)
        #     reward, new_state = environment.make_move(old_state, action)
        #     # new_state = environment.get_start_location()
        #
        #     if learn == True:  # Update Q-values if learning is specified
        #         agent.learn(old_state, reward, new_state, action)
        #
        #     cumulative_reward += reward
        #     step += 1
        #
        #     if environment.is_over(new_state):
        #         game_over = True

        old_state = environment.get_random_location()
        while step < max_steps_per_episode and game_over != True:
            action = agent.choose_action(environment.actions, old_state)
            reward, new_state = environment.make_move(old_state, action)

            if learn == True:
                agent.learn(old_state, reward, new_state, action)

            cumulative_reward += reward
            step += 1

            old_state = new_state

            if environment.is_over(new_state):
                if new_state in coords_snake_pits:
                    lost += 1
                if new_state in coords_rewards:
                    won += 1
                game_over = True

        reward_per_episode.append(cumulative_reward)

    return reward_per_episode, won, lost


def playSARSA(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    reward_per_episode = []
    won = 0
    lost = 0

    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False

        old_state = environment.get_random_location()
        action = agent.choose_action(environment.actions, old_state)

        while step < max_steps_per_episode and game_over != True:
            reward, new_state = environment.make_move(old_state, action)

            if learn == True:
                agent.learn(old_state, reward, new_state, action)

            next_action = agent.choose_action(environment.actions, new_state)

            cumulative_reward += reward
            step += 1

            old_state = new_state
            action = next_action

            if environment.is_over(new_state):
                if new_state in coords_snake_pits:
                    lost += 1
                if new_state in coords_rewards:
                    won += 1
                game_over = True

        reward_per_episode.append(cumulative_reward)

    return reward_per_episode, won, lost


dimensions = (9,9)
coords_walls = [(1,2), (1,3), (1,4), (1,5), (1,6), (2,6), (3,6), (4,6), (5,6), (7,1), (7,2), (7,3), (7,4)]
coords_snake_pits = [(6,5)]
coords_rewards = [(8,8)]
rewards = {0: -1, 1: -1, 2: -50, 3: 50}
actions = [(-1,0), (0,1), (1,0), (0,-1)]  # north east south west
policy = 1/4
grid = Gridworld(dimensions, coords_walls, coords_snake_pits, coords_rewards, rewards, actions, policy)

# monte carlo policy evaluation state value function
for _ in range(500):
    state = grid.get_random_location()
    grid.get_value_for_state(state)

# print(grid.reward_map)

heat_map = np.zeros(dimensions)
for i in range(dimensions[0]):
    for j in range(dimensions[1]):
        sum = grid.reward_map[i][j][0]
        nr = grid.reward_map[i][j][1]
        if nr != 0:
            heat_map[i][j] = sum//nr

print("\nState values matrix: ")
print(heat_map)
plt.imshow(heat_map, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()


agentQ = Q_Agent(grid)

reward_per_episode, won, lost = playQ(grid, agentQ, trials=100, max_steps_per_episode=500, learn=True)
print(reward_per_episode)
print(won)
print(lost)

print("\nQ table dictionary for Q learning")
print(agentQ.q_table)

route = np.zeros(dimensions, str)
print("\nQ table action matrix for Q learning")
for i in range(9):
    for j in range(9):
        q_values_of_state = agentQ.q_table[(i,j)]
        maxValue = max(q_values_of_state.values())
        if maxValue == 0:
            # print("0", end=" ")
            route[i][j] = "0"
        else:
            max_actions = [k for k, v in q_values_of_state.items() if v == maxValue]
            action = random.choice(max_actions)
            if action == (-1,0):
                # print("N", end=" ")
                route[i][j] = "N"
            if action == (1,0):
                # print("S", end=" ")
                route[i][j] = "S"
            if action == (0,-1):
                # print("W", end=" ")
                route[i][j] = "W"
            if action == (0, 1):
                route[i][j] = "E"
                # print("E", end=" ")
    # print("\n")
print(route)

# print("\nQ table value matrix for Q learning")
# for i in range(9):
#     for j in range(9):
#         q_values_of_state = agentQ.q_table[(i,j)]
#         maxValue = max(q_values_of_state.values())
#         print(int(maxValue), end=" ")
#     print("\n")

agentSARSA = Q_Agent(grid)
reward_per_episode2, won2, lost2 = playSARSA(grid, agentSARSA, trials=100, max_steps_per_episode=500, learn=True)
print(reward_per_episode2)
print(won2)
print(lost2)

print("\nQ table dictionary for SARSA learning")
print(agentSARSA.q_table)

route2 = np.zeros(dimensions, str)
print("\nQ table action matrix for SARSA learning")
for i in range(9):
    for j in range(9):
        q_values_of_state = agentSARSA.q_table[(i,j)]
        maxValue = max(q_values_of_state.values())
        if maxValue == 0:
            # print("0", end=" ")
            route2[i][j] = "0"
        else:
            max_actions = [k for k, v in q_values_of_state.items() if v == maxValue]
            action = random.choice(max_actions)
            if action == (-1,0):
                # print("N", end=" ")
                route2[i][j] = "N"
            if action == (1,0):
                # print("S", end=" ")
                route2[i][j] = "S"
            if action == (0,-1):
                # print("W", end=" ")
                route2[i][j] = "W"
            if action == (0, 1):
                route2[i][j] = "E"
                # print("E", end=" ")
    # print("\n")
print(route2)


# print("\nQ table value matrix for SARSA learning")
# for i in range(9):
#     for j in range(9):
#         q_values_of_state = agentSARSA.q_table[(i,j)]
#         maxValue = max(q_values_of_state.values())
#         print(int(maxValue), end=" ")
#     print("\n")

print("Q-learning mean: ", np.mean(reward_per_episode))
print("SARSA mean: ", np.mean(reward_per_episode2))

plt.plot(reward_per_episode, label='Q-learning')
plt.plot(reward_per_episode2, label='SARSA')
plt.title("Rewards per trials")
plt.legend()
plt.show()
