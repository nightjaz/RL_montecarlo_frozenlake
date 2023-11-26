import numpy as np
from frozen_lake_custom import FrozenLakeCustomEnv

def generate_episode(env):
    episode = []
    state = env.reset()
    if isinstance(state, tuple):
        state, _ = state

    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        print(next_state, reward, done, truncated, info)
        if isinstance(next_state, tuple):
            next_state, _ = next_state
        print(done)
        episode.append((state, action, reward))
        state = next_state
    return episode


def first_visit_monte_carlo(env, num_episodes, discount_factor=1.0):
    value_table = np.zeros(env.nS)
    returns = {s: [] for s in range(env.nS)}

    for _ in range(num_episodes):
        episode = generate_episode(env)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, _, reward = episode[t]
            G = discount_factor * G + reward
            if state not in [x[0] for x in episode[:t]]:
                returns[state].append(G)
                value_table[state] = np.mean(returns[state])

    return value_table

env = FrozenLakeCustomEnv(render_mode="human")
value_table = first_visit_monte_carlo(env, 10000)
print(value_table)

env.close()
