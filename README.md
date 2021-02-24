# RL

The program is implementing SARSA and Q-learning reinforcement learning algorithms for an agent to find the optimal paths in a simulated grid world (with walls, snake pits and treasures), which can be observed in the output. The agent is moving in the maze via actions and gets rewards.

The Monte Carlo policy evaluation computes the state value functions for the equiprobable policy (i.e. all 4 actions have probability 1/4), represented as a heat map (in the output) with the mean state values in every state.

SARSA (in combination with greedy policy) and Q-learning are used to search for an optimal policy. Both find optimal paths in the maze.
Relevant statistics are plotted to compare the two algorithms.
