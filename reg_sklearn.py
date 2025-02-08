import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


env_name = "Pendulum-v1"
# env_name = "Hopper-v4"
print("env_name:",env_name)

obs_log = np.load(env_name+"_obs_log.npy")
act_log = np.load(env_name+"_act_log.npy")
rew_log = np.load(env_name+"_rew_log.npy")
deltaobs_log = np.load(env_name+"_deltaobs_log.npy")

X = np.hstack([obs_log, act_log])
Y = np.hstack([rew_log, deltaobs_log])

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
print("X.shape:", X.shape, "Y.shape:", Y.shape)


from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(random_state=1, max_iter=200000, activation="relu", hidden_layer_sizes=(256,256), tol=1e-6, verbose=True)
reg.fit(X_train, y_train)

print("reg.coefs_\n",reg.coefs_)

y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)


output_dim = Y.shape[-1]
plot_col = 5
plot_row = int(output_dim // plot_col) + 1
fig, axs = plt.subplots(plot_row, plot_col, figsize=(20, 5*plot_row))
for i in range(output_dim):
    ir = int(i // plot_col)
    ic = i - plot_col*ir
    if plot_row>1:
        tmp_axs = axs[ir,ic]
    else:
        tmp_axs = axs[i]

    if i==0:
        outputname="rew"
    else:
        outputname="dobs"+str(i-1)

    tmp_axs.plot(y_train[:,i], y_pred_train[:,i], "o", label="train:"+outputname)
    tmp_axs.plot(y_test[:,i], y_pred_test[:,i], "o", label="test:"+outputname)
    tmp_axs.legend()
    
plt.show()
