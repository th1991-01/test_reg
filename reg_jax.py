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

    # grad_fn = jax.grad(loss_function)(reg_state.params)
    # grads = grad_fn(reg_state.params)

    value_and_grad_fn = jax.value_and_grad(loss_function)
    loss_value, grads = value_and_grad_fn(reg_state.params)

    reg_state = reg_state.apply_gradients(grads=grads)
    return reg_state, loss_value


for i in range(10000):
    reg_state, loss_value = update_params(reg_state, X_train, y_train)
    print(i,loss_value)

y_pred_train = reg.apply(reg_state.params, X_train)
y_pred_test = reg.apply(reg_state.params, X_test)


out_dim = Y.shape[-1]
fig, axs = plt.subplots(1, out_dim, figsize=(20, 5))
for i in range(out_dim):
    axs[i].plot(y_train[:,i], y_pred_train[:,i], "o", label="train")
    axs[i].plot(y_test[:,i], y_pred_test[:,i], "o", label="test")
plt.show()
