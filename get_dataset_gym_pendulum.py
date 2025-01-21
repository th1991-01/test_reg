import gymnasium as gym
print("gymnasium version:",gym.__version__)
import numpy as np

env = gym.make("Pendulum-v1", render_mode="rgb_array")

obs_log = []
act_log = []
rew_log = []
deltaobs_log = []

obs, info = env.reset(seed=42)
#imgs = [env.render()]
for _ in range(1000):
    act = env.action_space.sample()
    next_obs, rew, terminated, truncated, info = env.step(act)
    obs_log.append(obs)
    act_log.append(act)
    rew_log.append(rew)
    deltaobs_log.append(next_obs-obs)
    obs = next_obs[:]
    #imgs.append(env.render())

    if terminated or truncated:    
        observation, info = env.reset()

env.close()
obs_log = np.array(obs_log)
act_log = np.array(act_log)
rew_log = np.array(rew_log).reshape(-1,1)
deltaobs_log = np.array(deltaobs_log)

np.save("pendulum_obs_log.npy",obs_log)
np.save("pendulum_act_log.npy",act_log)
np.save("pendulum_rew_log.npy",rew_log)
np.save("pendulum_deltaobs_log.npy",deltaobs_log)

obs_log = np.load("pendulum_obs_log.npy")
act_log = np.load("pendulum_act_log.npy")
rew_log = np.load("pendulum_rew_log.npy")
deltaobs_log = np.load("pendulum_deltaobs_log.npy")


print(obs_log.shape, act_log.shape, rew_log.shape)
