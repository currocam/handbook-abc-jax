import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpyro
    import numpyro.distributions as dist
    import numpy as np
    import arviz as az

    import seaborn as sns
    import matplotlib.pyplot as plt
    import arviz_plots as azp

    azp.style.use("arviz-vibrant")
    return az, dist, jax, jnp, jr, mo, np, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #  ABC Rejection Sampling

    ## References

    [Chapter 1: Overview of Approximate Bayesian Computation](https://arxiv.org/pdf/1802.09720)

    ## Algorithm

    **Inputs:**
    - Prior distribution distribution $\pi(\theta)$
    - Procedure for generating data under the model $p(y|\theta)$
    - Proposal density $g(\theta)$ with $g(\theta) > 0$ if $\pi(\theta|y_{obs}) > 0$
    - Number of samples $N > 0$
    - Kernel function $K_h(u)$ with bandwidth $h > 0$
    - Summary statistic $s = S(y)$
    - Observed summary statistic $s_{obs} = S(y_{obs})$

    **Algorithm:**

    For $i = 1, \ldots, N$:
    1. Generate $\theta^{(i)} \sim g(\theta)$ from the proposal density
    2. Generate $y \sim p(y|\theta^{(i)})$ from the likelihood
    3. Compute summary statistic $s = S(y)$
    4. Accept $\theta^{(i)}$ with probability

       $\displaystyle \frac{K_h(\|s - s_{obs}\|) \pi(\theta^{(i)})}{K \cdot g(\theta^{(i)})}$

       Otherwise, return to step 1

    **Output:**
    - Parameter vectors $\theta^{(1)}, \ldots, \theta^{(N)} \sim \pi_{ABC}(\theta|s_{obs})$

    ## Popular kernel functions
    """)
    return


@app.cell
def _(jnp):
    def uniform_kernel(u):
        """Uniform kernel: 1/2 * I(|u| ≤ 1)"""
        return jnp.where(jnp.abs(u) <= 1, 0.5, 0.0)

    def triangular_kernel(u):
        """Triangular kernel: (1 - |u|) * I(|u| ≤ 1)"""
        return jnp.where(jnp.abs(u) <= 1, 1 - jnp.abs(u), 0.0)

    def epanechnikov_kernel(u):
        """Epanechnikov kernel: 3/4 * (1 - u²) * I(|u| ≤ 1)"""
        return jnp.where(jnp.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)

    def biweight_kernel(u):
        """Biweight kernel: 15/16 * (1 - u²)³ * I(|u| ≤ 1)"""
        return jnp.where(jnp.abs(u) <= 1, (15 / 16) * (1 - u**2) ** 3, 0.0)

    def gaussian_kernel(u):
        """Gaussian kernel: 1/√(2π) * exp(-u²/2)"""
        return (1 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * u**2)

    return (
        biweight_kernel,
        epanechnikov_kernel,
        gaussian_kernel,
        triangular_kernel,
        uniform_kernel,
    )


@app.cell
def _(
    biweight_kernel,
    epanechnikov_kernel,
    gaussian_kernel,
    jax,
    jnp,
    plt,
    triangular_kernel,
    uniform_kernel,
):
    u_values = jnp.linspace(-2, 2, 400)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(u_values, jax.vmap(uniform_kernel)(u_values), label="Uniform", linewidth=2)
    ax.plot(
        u_values, jax.vmap(triangular_kernel)(u_values), label="Triangular", linewidth=2
    )
    ax.plot(
        u_values,
        jax.vmap(epanechnikov_kernel)(u_values),
        label="Epanechnikov",
        linewidth=2,
    )
    ax.plot(
        u_values, jax.vmap(biweight_kernel)(u_values), label="Biweight", linewidth=2
    )
    ax.plot(
        u_values, jax.vmap(gaussian_kernel)(u_values), label="Gaussian", linewidth=2
    )

    ax.set_xlabel("u", fontsize=12)
    ax.set_ylabel("K(u)", fontsize=12)
    ax.set_title("ABC Kernel Functions (Figure 1.3)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(az, gaussian_kernel, jax, jnp, jr, np):
    def abc_rejection_sample(
        key,
        log_prior_density,
        log_proposal_density,
        proposal_sample,
        simulator,
        num_candidates,
        s_obs,
        distance,
        summary_stat=jax.nn.identity,
        kernel=gaussian_kernel,
        h=1.0,
    ):
        """
        ABC rejection sampling using parallel sampling strategy.

        Args:
            key: JAX random key
            log_prior_density: function log π(θ) computing log prior density
            log_proposal_density: function log g(θ) computing log proposal density
            proposal_sample: function to sample from proposal (key, n) -> dict of arrays
            simulator: function p(y|θ) to sample data (key, params) -> y
            num_candidates: number of candidates to generate
            s_obs: observed summary statistics
            summary_stat: function S(y) computing summary statistics
            distance: distance function for summary statistics
            kernel: kernel function Kh(u)
            h: bandwidth parameter for kernel

        Returns:
            az.InferenceData with accepted parameter samples in the posterior group
        """
        key, key_propose, key_likelihood, key_accept = jr.split(key, 4)

        # Step 1: Generate candidates from proposal (dict of arrays)
        theta_candidates = proposal_sample(key_propose, num_candidates)

        # Step 2 & 3: Generate data and compute summary statistics
        keys_likelihood = jr.split(key_likelihood, num_candidates)
        y_samples = jax.vmap(simulator)(keys_likelihood, theta_candidates)
        s_samples = jax.vmap(lambda y: jnp.atleast_1d(summary_stat(y)))(y_samples)
        s_obs_vec = jnp.atleast_1d(s_obs)

        # Step 4: Compute acceptance probabilities
        distances = distance(s_samples, s_obs_vec)
        kernel_values = jax.vmap(kernel)(distances / h)

        log_prior_values = jax.vmap(log_prior_density)(theta_candidates)
        log_proposal_values = jax.vmap(log_proposal_density)(theta_candidates)
        max_log_ratio = jnp.max(log_prior_values - log_proposal_values)

        # Accept with probability: Kh(||s - s_obs||) * π(θ) / (K * g(θ))
        log_acceptance_prob = (
            jnp.log(kernel_values)
            + log_prior_values
            - max_log_ratio
            - log_proposal_values
        )
        acceptance_prob = jnp.exp(jnp.minimum(log_acceptance_prob, 0))

        accepted = jr.uniform(key_accept, (num_candidates,)) <= acceptance_prob
        accepted_samples = jax.tree.map(lambda arr: arr[accepted], theta_candidates)

        # ArviZ expects shape (chains, draws, *param_shape)
        posterior_dict = {
            k: np.asarray(v)[np.newaxis, :] for k, v in accepted_samples.items()
        }
        idata = az.from_dict(posterior=posterior_dict)
        idata.add_groups(
            {
                "prior": {
                    k: np.asarray(v)[np.newaxis, :] for k, v in theta_candidates.items()
                }
            }
        )
        return idata

    return (abc_rejection_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example (from the book)

    Suppose the model is specified as $y_1, \dots, y_{50} \sim N(\theta, 1)$, with uniform prior $\theta \sim U(-5, 5)$.
    """)
    return


@app.cell
def _(dist, jr):
    theta_true = 2.5
    observed_y = dist.Normal(theta_true, 1).sample(jr.key(1), (50,))
    observed_y
    return (observed_y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rejection (with sufficient summary statistics)
    """)
    return


@app.cell
def _(dist, jnp):
    # Define prior distribution
    prior = dist.Uniform(-5, 5)
    # Often, we use the prior as the proposal distribution
    log_prior_density = lambda params: prior.log_prob(params["theta"])
    log_proposal_density = lambda params: prior.log_prob(params["theta"])
    proposal_sample = lambda key, n: {"theta": prior.sample(key, (n,))}
    # Sufficient summary statistic
    summary_stat = lambda y: jnp.mean(y)
    # Bandwidth for kernel and number of candidates to generate
    num_candidates = 50_000
    return (
        log_prior_density,
        log_proposal_density,
        num_candidates,
        proposal_sample,
        summary_stat,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Often, the simulator function cannot be JIT compiled and it's expensive to evaluate. Such code must be wrapped with `pure_callback`.
    """)
    return


@app.cell
def _(jax, jnp):
    def black_box_simulator(key, params):
        import numpy as np

        # Convert JAX random key to numpy seed by extracting bytes
        # The key is a JAX array, extract first integer from it
        key_array = np.asarray(key)
        seed = int(key_array.view(np.uint32).flat[0])
        np.random.seed(seed)
        n_obs = 50
        # Extract scalar value from params["theta"] (handles both scalars and 1D arrays)
        theta_val = float(np.squeeze(np.asarray(params["theta"])))
        y = np.random.normal(theta_val, 1.0, n_obs)
        return jnp.array(y)

    # Wrap with pure_callback to make it compatible with JAX transformations
    # The callback is called with arguments inside a vmap loop
    def simulator(key, params):
        return jax.pure_callback(
            black_box_simulator,
            jax.ShapeDtypeStruct(shape=(50,), dtype=jnp.float32),
            key,
            params,
            vmap_method="sequential",
        )

    return (simulator,)


@app.cell
def _(
    abc_rejection_sample,
    gaussian_kernel,
    jnp,
    jr,
    log_prior_density,
    log_proposal_density,
    num_candidates,
    observed_y,
    proposal_sample,
    simulator,
    summary_stat,
):
    # Compute observed summary statistics
    s_obs = jnp.atleast_1d(summary_stat(observed_y))
    posterior_abc_sf = abc_rejection_sample(
        key=jr.key(0),  # Set seed for reproducibility
        log_prior_density=log_prior_density,
        log_proposal_density=log_proposal_density,
        proposal_sample=proposal_sample,
        simulator=simulator,
        num_candidates=num_candidates,
        s_obs=s_obs,
        # Euclidean distance for summary statistics
        distance=lambda s, s_obs: jnp.linalg.norm(s - s_obs, axis=-1),
        summary_stat=summary_stat,
        kernel=gaussian_kernel,
        h=0.1,
    )
    return (posterior_abc_sf,)


@app.cell
def _(az, posterior_abc_sf):
    az.summary(posterior_abc_sf)
    return


@app.cell
def _(dist, jnp, jr, observed_y, plt, posterior_abc_sf, sns):
    sns.kdeplot(
        posterior_abc_sf.posterior["theta"].values.flatten(),
        alpha=0.6,
        color="C0",
        label="ABC Posterior",
    )
    # Ground truth posterior (conjugate normal)
    posterior_exact = dist.Normal(observed_y.mean(), 1 / jnp.sqrt(50)).sample(
        jr.key(1), (5000,)
    )
    posterior_exact = posterior_exact[
        (posterior_exact >= -5) & (posterior_exact <= 5)
    ]  # Truncate to match uniform prior support
    sns.kdeplot(posterior_exact, alpha=0.6, color="C1", label="Exact Posterior")
    plt.legend()
    plt.gca()
    return (posterior_exact,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Often, sufficient summary statistics are not available. One popular distance metric between the observed and simulated data is the Wasserstein distance:
    """)
    return


@app.cell
def _(jax, jnp):
    def wasserstein_distance(a, b):
        from scipy.stats import wasserstein_distance

        return jax.pure_callback(
            lambda a, b: jnp.array([wasserstein_distance(x, b) for x in a]),
            jax.ShapeDtypeStruct((a.shape[0],), jnp.float32),
            a,
            b,
            vmap_method="sequential",
        )

    return (wasserstein_distance,)


@app.cell
def _(
    abc_rejection_sample,
    gaussian_kernel,
    jr,
    log_prior_density,
    log_proposal_density,
    num_candidates,
    observed_y,
    proposal_sample,
    simulator,
    wasserstein_distance,
):
    posterior_abc_ws = abc_rejection_sample(
        key=jr.key(0),  # Set seed for reproducibility
        log_prior_density=log_prior_density,
        log_proposal_density=log_proposal_density,
        proposal_sample=proposal_sample,
        simulator=simulator,
        num_candidates=num_candidates * 2,
        s_obs=observed_y,  # Use raw observed data
        kernel=gaussian_kernel,
        distance=wasserstein_distance,
        h=0.1,
    )
    return (posterior_abc_ws,)


@app.cell
def _(az, posterior_abc_ws):
    az.summary(posterior_abc_ws)
    return


@app.cell
def _(plt, posterior_abc_ws, posterior_exact, sns):
    sns.kdeplot(
        posterior_abc_ws.posterior["theta"].values.flatten(),
        alpha=0.6,
        color="C0",
        label="ABC Posterior (Wasserstein distance)",
    )
    # Ground truth posterior (conjugate normal)
    sns.kdeplot(posterior_exact, alpha=0.6, color="C1", label="Exact Posterior")
    plt.legend()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
