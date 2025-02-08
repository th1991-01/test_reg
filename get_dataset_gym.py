import gymnasium as gym
print("gymnasium version:",gym.__version__)
import numpy as np

# env_name = "Pendulum-v1"
env_name = "Hopper-v4"
print("env_name:",env_name)

env = gym.make(env_name, render_mode="rgb_array")

obs_log = []
act_log = []
rew_log = []
deltaobs_log = []
datanum = 10000

obs, info = env.reset(seed=42)
episode_count = 0
for i in range(datanum):
    act = env.action_space.sample()
    next_obs, rew, terminated, truncated, info = env.step(act)
    obs_log.append(obs)
    act_log.append(act)
    rew_log.append(rew)
    deltaobs_log.append(next_obs-obs)
    obs = next_obs[:]

    if terminated or truncated:
        episode_count += 1
        observation, info = env.reset()

env.close()
obs_log = np.array(obs_log)
act_log = np.array(act_log)
rew_log = np.array(rew_log).reshape(-1,1)
deltaobs_log = np.array(deltaobs_log)

np.save(env_name+"_obs_log.npy",obs_log)
np.save(env_name+"_act_log.npy",act_log)
np.save(env_name+"_rew_log.npy",rew_log)
np.save(env_name+"_deltaobs_log.npy",deltaobs_log)

obs_log = np.load(env_name+"_obs_log.npy")
act_log = np.load(env_name+"_act_log.npy")
rew_log = np.load(env_name+"_rew_log.npy")
deltaobs_log = np.load(env_name+"_deltaobs_log.npy")


print("obs_log.shape:", obs_log.shape, "act_log.shape:", act_log.shape, "rew_log.shape:", rew_log.shape)
print("episode_count:",episode_count)
