import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import arviz as az
    import arviz_plots as azp
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist

    from emplik import calc_log_ael

    azp.style.use("arviz-vibrant")
    return az, calc_log_ael, dist, jax, jnp, jr, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Empirical likelihoods NUTS

    ## References

    [Approximating the Likelihood in Approximate Bayesian Computation](https://arxiv.org/pdf/1803.06645)


    Empirical likelihoods can be used to do approximate statistical inference without making strong parametric assumptions about the data generating process. Rather than assuming that a certain distribution (e.g., normal, Poisson) generated the data, we can specify moment conditions that the data should satisfy (e.g., mean) and use empirical likelihoods to perform inference based on these moments.

    This idea was introduced in the late 1980s. The appealing thing is that the empirical likelihood shares some assymptotic properties with the actual likelihood. However, it's not widely used (I think), I wonder why. In this notebook I use the log-adjusted empirical likelihood following the jax implementation in [https://github.com/weiyaw/epel/blob/main/emplik.py](https://github.com/weiyaw/epel/blob/main/emplik.py).
    """)
    return


@app.cell
def _(calc_log_ael, jax, jnp):
    def h_poisson(z, theta):
        return jnp.atleast_1d(z - theta[0])

    @jax.jit
    def adjusted_empirical_loglikelihood(data, theta):
        a_n = 0.5 * jnp.log(data.shape[0])
        return calc_log_ael(h_poisson, data, theta, a_n, check_sum_to_1=False)

    return (adjusted_empirical_loglikelihood,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example

    Suppose the model is specified as $y_1, \dots, y_{50} \sim \text{Poisson}(\theta)$, with uniform prior $\theta \sim U(0, 10)$.
    """)
    return


@app.cell
def _(dist, jnp, jr):
    theta_true = 2.5
    observed_y = dist.Poisson(theta_true).sample(jr.key(1), (50,)).astype(jnp.float32)
    observed_y
    return (observed_y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Empirical likelihood versus true likelihood
    """)
    return


@app.cell
def _(jnp, observed_y):
    obs_moments = jnp.array([observed_y.mean()])
    return


@app.cell
def _(adjusted_empirical_loglikelihood, dist, jax, jnp, np, observed_y, plt):
    # Define function to compute empirical likelihood for a single theta value
    def em_loglik_single(theta_val):
        return adjusted_empirical_loglikelihood(
            observed_y, jnp.array([theta_val])
        ).sum()

    # Compute gradients using JAX autodiff
    grad_em_loglik = jax.grad(em_loglik_single)

    # Evaluate gradients on grid
    _grid = np.linspace(1, 6.9, 50)
    _em_grads = np.array([float(grad_em_loglik(jnp.array(theta))) for theta in _grid])

    # True likelihood gradient for comparison
    def true_loglik_single(theta_val):
        return dist.Poisson(theta_val).log_prob(observed_y).sum()

    grad_true_loglik = jax.grad(true_loglik_single)
    _true_grads = np.array(
        [float(grad_true_loglik(jnp.array(theta))) for theta in _grid]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot log-likelihoods
    _liks = jax.vmap(dist.Poisson(_grid).log_prob)(observed_y).sum(axis=0)
    _liks = _liks - _liks.max()
    _em_liks = jax.vmap(
        lambda theta: adjusted_empirical_loglikelihood(
            observed_y, jnp.array([theta])
        ).sum()
    )(_grid)
    _em_liks = _em_liks - _em_liks.max()

    ax1.plot(_grid, _liks, label="True log-likelihood", linewidth=2)
    ax1.plot(_grid, _em_liks, label="Empirical log-likelihood", linewidth=2)
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel("log-likelihood")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot gradients
    ax2.plot(_grid, _true_grads, label="True gradient", linewidth=2)
    ax2.plot(_grid, _em_grads, label="Empirical gradient", linewidth=2)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\frac{d}{d\theta} \log p(\theta | y)$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.gca()
    return


@app.cell
def _(adjusted_empirical_loglikelihood, dist, jax, jnp, observed_y):
    prior = dist.Uniform(0, 10, validate_args=True)

    def prior_log_prob(params):
        theta = params["theta"]
        return prior.log_prob(theta)

    def loglikelihood(params):
        theta = params["theta"]
        return adjusted_empirical_loglikelihood(observed_y, jnp.array([theta])).sum()

    def logdensity(params):
        lprior = prior_log_prob(params)
        ll = jax.lax.cond(
            jnp.isfinite(lprior).squeeze(),
            lambda _: loglikelihood(params),
            lambda _: -jnp.inf,
            operand=lprior,
        )
        return jnp.squeeze(lprior + ll)

    return (logdensity,)


@app.cell
def _(az, jax, jnp, logdensity, np):
    import blackjax

    def empirical_likelihood_nuts(
        key, log_density, initial_positions, num_samples, num_warmups=1000
    ):
        """
        key: JAX random key
        log_density: log posterior density
        initial_positions: list of initial position dicts, one per chain
        num_samples : number of samples
        num_warmups: number of warmups iterations
        """

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                # Extract theta from position dict
                theta = state.position["theta"]
                return state, theta

            keys = jax.random.split(rng_key, num_samples)
            _, theta_samples = jax.lax.scan(one_step, initial_state, keys)

            return theta_samples

        def _run_chain(args):
            init_pos, chain_key = args
            warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
            warmup_key, sample_key = jax.random.split(chain_key, 2)
            (state, parameters), _ = warmup.run(
                warmup_key, init_pos, num_steps=num_warmups
            )
            kernel = blackjax.nuts(logdensity, **parameters).step
            return inference_loop(sample_key, kernel, state, num_samples)

        num_chains = len(initial_positions)
        chain_keys = jax.random.split(key, num_chains)

        # Stack initial positions
        stacked_init = jax.tree.map(
            lambda *arrays: jnp.stack(arrays, axis=0), *initial_positions
        )

        # Run chains sequentially
        all_theta_samples = []
        for i in range(num_chains):
            init_pos = {k: v[i] for k, v in stacked_init.items()}
            theta_samples = _run_chain((init_pos, chain_keys[i]))
            all_theta_samples.append(theta_samples)

        # Stack across chains: shape (num_chains, num_samples, param_dim)
        all_theta = jnp.stack(all_theta_samples, axis=0)
        posterior_dict = {"theta": np.asarray(all_theta)}
        return az.from_dict(posterior=posterior_dict)

    return (empirical_likelihood_nuts,)


@app.cell
def _(jnp, jr):
    initial_positions = [
        {"theta": jnp.array([1.0])},
        {"theta": jnp.array([2.0])},
        {"theta": jnp.array([3.0])},
        {"theta": jnp.array([4.0])},
    ]
    nuts_key = jr.key(0)
    return initial_positions, nuts_key


@app.cell
def _(empirical_likelihood_nuts, initial_positions, logdensity, nuts_key):
    idata = empirical_likelihood_nuts(
        nuts_key, logdensity, initial_positions, 2000, num_warmups=2000
    )
    return (idata,)


@app.cell
def _(az, idata):
    az.summary(idata)
    return


@app.cell
def _(az, idata):
    az.plot_trace(idata)
    return


if __name__ == "__main__":
    app.run()
