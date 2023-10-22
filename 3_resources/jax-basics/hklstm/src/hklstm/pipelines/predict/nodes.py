"""
This is a boilerplate pipeline 'predict'
generated using Kedro 0.18.4
"""
import warnings

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotnine as gg

from ..data_process.nodes import Dataset

gg.theme_set(gg.theme_bw())
warnings.filterwarnings("ignore")


def fast_autoregressive_predict_fn(context, seq_len):
    """Given a context, autoregressively generate the rest of a sine wave."""
    core = hk.LSTM(32)
    dense = hk.Linear(1)
    state = core.initial_state(context.shape[1])
    # Unroll over the context using `hk.dynamic_unroll`.
    # As before, we `hk.BatchApply` the Linear for efficiency.
    context_outs, state = hk.dynamic_unroll(core, context, state)
    context_outs = hk.BatchApply(dense)(context_outs)

    # Now, unroll one step at a time using the running recurrent state.
    ar_outs = []
    x = context_outs[-1]
    for _ in range(seq_len - context.shape[0]):
        x, state = core(x, state)
        x = dense(x)
        ar_outs.append(x)
    return jnp.concatenate([context_outs, jnp.stack(ar_outs)])


def plot_samples(truth: np.ndarray, prediction: np.ndarray) -> gg.ggplot:
    assert truth.shape == prediction.shape
    df = pd.DataFrame(
        {"truth": truth.squeeze(), "predicted": prediction.squeeze()}
    ).reset_index()
    df = pd.melt(df, id_vars=["index"], value_vars=["truth", "predicted"])
    plot = (
        gg.ggplot(df) + gg.aes(x="index", y="value", color="variable") + gg.geom_line()
    )
    return plot


def plot_predictions(trained_params, valid_ds, SEQ_LEN):
    sample_x, _ = next(valid_ds)
    context_length = SEQ_LEN // 8
    # Cut the batch-size 1 context from the start of the sequence.
    context = sample_x[:context_length, :1]

    # We can reuse params we got from training for inference - as long as the
    # declaration order is the same.
    fast_ar_predict = hk.transform(fast_autoregressive_predict_fn)
    fast_ar_predict = jax.jit(fast_ar_predict.apply, static_argnums=3)
    # Reuse the same context from the previous cell.
    predicted = fast_ar_predict(trained_params, None, context, SEQ_LEN)
    # The plots should be equivalent!
    plot = plot_samples(sample_x[1:, :1], predicted[:-1])
    plot += gg.geom_vline(xintercept=len(context), linetype="dashed")

    plot.save("data/08_reporting/output_plot.png")
    return plot
