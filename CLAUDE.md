# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a handbook implementing **Approximate Bayesian Computation (ABC)** algorithms using JAX, NumPyro, BlackJAX, and ArviZ. Each notebook covers a different ABC variant: rejection sampling, importance sampling, MCMC, SMC, and stopping criteria.

## Commands

```bash
# Format all notebooks
make format        # runs: uvx ruff format

# Check notebooks for errors
make check         # runs: uvx marimo check

# Export all .py marimo notebooks to .ipynb
make export-all    # runs: uv run marimo export ipynb for each notebooks/*.py

# Run all steps
make all

# Run a single marimo notebook interactively
uv run marimo edit notebooks/<name>.py

# Export a single notebook
uv run marimo export ipynb notebooks/<name>.py -o notebooks/<name>.ipynb
```

## Architecture

**Notebooks are authored as marimo `.py` files** (in `notebooks/`). The `.ipynb` files are derived artifacts exported from them — do not edit the `.ipynb` files directly.

Each `.py` notebook follows the marimo app pattern:
```python
app = marimo.App(width="medium")

@app.cell
def _(deps...):
    ...
    return (outputs,)
```

### Key patterns across notebooks

- **JAX-first**: All numerical code uses `jax.numpy` and `jax.random` (via `jnp`/`jr`). Use `jax.vmap` for vectorized simulation over candidates.
- **Non-JAX simulators**: When a simulator can't be JIT-compiled (e.g., uses NumPy/SciPy internals), wrap it with `jax.pure_callback(..., vmap_method="sequential")`.
- **ArviZ for results**: Accepted samples are returned as `az.InferenceData` objects with a `posterior` group (shape `(chains, draws, *param_shape)`). Parameters are passed as dicts of arrays.
- **Distance functions**: Take `(s_samples, s_obs)` where `s_samples` is `(N, d)` and return `(N,)` distances. Euclidean (`jnp.linalg.norm`) and Wasserstein (via `scipy`, wrapped with `pure_callback`) are used.

### Dependencies

Managed with `uv`. Key packages: `jax`, `numpyro`, `blackjax`, `arviz`, `arviz-plots`, `marimo`, `msprime`, `matplotlib`, `seaborn`.
