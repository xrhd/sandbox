"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.4
"""


import haiku as hk
import jax
import jax.numpy as jnp
import optax

from ..data_process.nodes import Dataset


def unroll_net(seqs: jnp.ndarray):
    """Unrolls an LSTM over seqs, mapping each output to a scalar."""
    # seqs is [T, B, F].
    core = hk.LSTM(32)
    batch_size = seqs.shape[1]
    outs, state = hk.dynamic_unroll(core, seqs, core.initial_state(batch_size))
    # We could include this Linear as part of the recurrent core!
    # However, it's more efficient on modern accelerators to run the linear once
    # over the entire sequence than once per sequence element.
    return hk.BatchApply(hk.Linear(1))(outs), state


model = hk.transform(unroll_net)


def train_model(train_ds: Dataset, valid_ds: Dataset) -> hk.Params:
    """Initializes and trains a model on train_ds, returning the final params."""
    rng = jax.random.PRNGKey(428)
    opt = optax.adam(1e-3)

    @jax.jit
    def loss(params, x, y):
        pred, _ = model.apply(params, None, x)
        return jnp.mean(jnp.square(pred - y))

    @jax.jit
    def update(step, params, opt_state, x, y):
        l, grads = jax.value_and_grad(loss)(params, x, y)
        grads, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return l, params, opt_state

    # Initialize state.
    sample_x, _ = next(train_ds)
    params = model.init(rng, sample_x)
    opt_state = opt.init(params)

    for step in range(2001):
        if step % 100 == 0:
            x, y = next(valid_ds)
            print("Step {}: valid loss {}".format(step, loss(params, x, y)))

        x, y = next(train_ds)
        train_loss, params, opt_state = update(step, params, opt_state, x, y)
        if step % 100 == 0:
            print("Step {}: train loss {}".format(step, train_loss))

    return params
