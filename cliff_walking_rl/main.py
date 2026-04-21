import numpy as np
import matplotlib.pyplot as plt


#  Epsilon-Greedy Policy
def choose_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    r, c = state
    return np.argmax(Q[r, c])


#  Environment
class CliffWalking:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state

        if action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c -= 1
        elif action == 3: c += 1

        r = max(0, min(r, self.rows - 1))
        c = max(0, min(c, self.cols - 1))

        next_state = (r, c)

        #  Cliff
        if r == 3 and 1 <= c <= 10:
            return self.start, -100, False

        #  Goal
        if next_state == self.goal:
            return next_state, 0, True

        #  Reward shaping (push toward goal)
        reward = -1 - 0.01 * abs(c - 11)
        return next_state, reward, False


#  Q-Learning
def q_learning(env, episodes=3000, alpha=0.5, gamma=0.9):
    Q = np.zeros((4, 12, 4))
    rewards = []

    for ep in range(episodes):
        epsilon = max(0.01, 0.1 * (0.99 ** ep))
        state = env.reset()
        total_reward = 0

        for _ in range(200):
            action = choose_action(state, Q, epsilon)
            next_state, reward, done = env.step(action)

            r, c = state
            nr, nc = next_state

            Q[r, c, action] += alpha * (
                reward + gamma * np.max(Q[nr, nc]) - Q[r, c, action]
            )

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    return Q, rewards


#  SARSA
def sarsa(env, episodes=3000, alpha=0.5, gamma=0.9):
    Q = np.zeros((4, 12, 4))
    rewards = []

    for ep in range(episodes):
        epsilon = max(0.01, 0.1 * (0.99 ** ep))
        state = env.reset()
        action = choose_action(state, Q, epsilon)
        total_reward = 0

        for _ in range(200):
            next_state, reward, done = env.step(action)
            next_action = choose_action(next_state, Q, epsilon)

            r, c = state
            nr, nc = next_state

            Q[r, c, action] += alpha * (
                reward + gamma * Q[nr, nc, next_action] - Q[r, c, action]
            )

            state = next_state
            action = next_action
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    return Q, rewards


#  Policy Visualization
def print_policy(Q):
    directions = ['↑', '↓', '←', '→']

    for i in range(4):
        for j in range(12):
            if (i, j) == (3, 0):
                print('S', end=' ')
            elif (i, j) == (3, 11):
                print('G', end=' ')
            elif i == 3 and 1 <= j <= 10:
                print('C', end=' ')
            else:
                best = np.argmax(Q[i, j])
                print(directions[best], end=' ')
        print()


#  MAIN
if __name__ == "__main__":
    print("TRAINING STARTED...\n")

    env = CliffWalking()

    Q_q, rewards_q = q_learning(env)
    Q_s, rewards_s = sarsa(env)

    print("TRAINING COMPLETE!\n")

    print("Q-Learning Last Rewards:", rewards_q[-10:])
    print("SARSA Last Rewards:", rewards_s[-10:])

    #   Plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_q, label="Q-Learning")
    plt.plot(rewards_s, label="SARSA")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning vs SARSA (Cliff Walking)")
    plt.legend()
    plt.grid()
    plt.show()

    #  Policies
    print("\nQ-Learning Policy:")
    print_policy(Q_q)

    print("\nSARSA Policy:")
    print_policy(Q_s)