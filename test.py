from frozen_lake_custom import FrozenLakeCustomEnv
import random

Env = FrozenLakeCustomEnv(render_mode="human")

print(Env.P)

actions = [0, 1, 2, 3]

observation, info = Env.reset()

for _ in range(1000):
    action = random.choice(actions)
    observation, reward, terminated, truncated, info = Env.step(action)
    print(reward)

    if terminated or truncated:
        observation, info = Env.reset()

Env.close()