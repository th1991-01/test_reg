import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# env_name = "Pendulum-v1"
env_name = "Hopper-v4"
print("env_name:",env_name)

obs_log = np.load(env_name+"_obs_log.npy")
act_log = np.load(env_name+"_act_log.npy")
rew_log = np.load(env_name+"_rew_log.npy")
deltaobs_log = np.load(env_name+"_deltaobs_log.npy")
print("obs_log.shape:", obs_log.shape, "\nact_log.shape:", act_log.shape, "\nrew_log.shape:", rew_log.shape, "\ndeltaobs_log.shape:", deltaobs_log.shape)

X = np.hstack([obs_log, act_log])
Y = np.hstack([rew_log, deltaobs_log])
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

print("X_train.shape:", X_train.shape, "X_test.shape:", X_test.shape, "\ny_train.shape:", y_train.shape, "y_test.shape:", y_test.shape)


import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

key = jax.random.PRNGKey(42)

class RegModel(nn.Module):
    output_dim: int
    @nn.compact
    def __call__(self, x_input):
        x = nn.Dense(256)(x_input)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        return nn.Dense(self.output_dim)(x)

reg = RegModel(y_train.shape[-1])
reg.apply = jax.jit(reg.apply)

reg_state = TrainState.create(
    apply_fn = reg.apply,
    params = reg.init(key, jnp.zeros(X_train.shape[-1])),
    tx = optax.adam(learning_rate=1e-3)
)

@jax.jit
def update_params(reg_state, X, Y):

    def loss_function(params):
        pred = reg.apply(params, X)
        loss = jnp.mean((pred - Y)**2)
        return loss

    value_and_grad_fn = jax.value_and_grad(loss_function)
    loss_value, grads = value_and_grad_fn(reg_state.params)

    reg_state = reg_state.apply_gradients(grads=grads)
    return reg_state, loss_value


for i in range(10000):
    reg_state, loss_value = update_params(reg_state, X_train, y_train)
    print(i,loss_value)

y_pred_train = reg.apply(reg_state.params, X_train)
y_pred_test = reg.apply(reg_state.params, X_test)


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
