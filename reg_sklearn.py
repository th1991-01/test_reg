import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


obs_log = np.load("pendulum_obs_log.npy")
act_log = np.load("pendulum_act_log.npy")
rew_log = np.load("pendulum_rew_log.npy")
deltaobs_log = np.load("pendulum_deltaobs_log.npy")

X = np.hstack([obs_log, act_log])
Y = np.hstack([rew_log, deltaobs_log])

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)



from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(random_state=1, max_iter=200000, activation="relu", hidden_layer_sizes=(256,256), tol=1e-6, verbose=True)
reg.fit(X_train, y_train)

print("reg.coefs_\n",reg.coefs_)

y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)


out_dim = Y.shape[-1]
fig, axs = plt.subplots(1, out_dim, figsize=(20, 5))
for i in range(out_dim):
    axs[i].plot(y_train[:,i], y_pred_train[:,i], "o", label="train")
    axs[i].plot(y_test[:,i], y_pred_test[:,i], "o", label="test")
plt.show()
