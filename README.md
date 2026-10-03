# Dirichlet Mixed-Membership Model (DMMM)

This package provides the Julia estimation and prediction machinery for a
Dirichlet Mixed-Membership Model designed for multivariate distributional
credibility. Each policyholder has a fixed latent composition drawn from a
Dirichlet distribution; period-specific risk classes are drawn from that
composition and responses follow class-specific, support-appropriate expert
distributions.

## Capabilities

- exact and Monte Carlo DMMM E-steps;
- direct enumeration, a closed-form grouping method for histories of at most
  three periods, count-state dynamic programming, and collapsed Gibbs MCEM;
- ECM fitting of the multinomial-logit gate, total concentration, and experts;
- optional fixed positive response-component weights for a generalized
  posterior/weighted response criterion;
- prior class probabilities, period responsibilities, posterior composition,
  credibility weights, predictive means/variances, and finite predictive
  mixtures;
- generic seeded simulation from user-supplied DGP settings.

## Installation

From a Julia REPL, activate the local staged package and instantiate it:

```julia
using Pkg
Pkg.activate("path/to/DMMM")
Pkg.instantiate()
using DMMM
```

The package keeps its existing UUID. A `Manifest.toml` is intentionally not
included; dependency versions are constrained in `Project.toml` and can be
resolved in the user's environment.

## Minimal synthetic example

```julia
using DMMM, Random

group = repeat(1:20; inner=3)
X = ones(length(group), 1)
alpha = reshape([log(0.6 / 0.4), 0.0], 2, 1)
experts = reshape(Any[PoissonExpert(1.0), PoissonExpert(5.0)], 1, 2)

sample = simulate_dmmm(
    alpha, 3.0, X, group, experts;
    rng=MersenneTwister(2026), return_latent=true,
)

fit = fit_DMMM(
    sample.Y, X, group, zeros(2, 1), 2.0,
    reshape(Any[PoissonExpert(1.3), PoissonExpert(4.0)], 1, 2);
    estep=:closed_form, exact_update=:block, penalty=false, print_steps=0,
)

posterior = predict_class_posterior_dmmm(sample.Y, X, group, fit)
future_mean = predict_mean_posterior_dmmm(sample.Y, X, group, fit)
future_laws = predictive_mixture_dmmm(sample.Y, X, group, fit)
```

`alpha` uses the final class as the zero-coefficient baseline. `phi` is the
positive total Dirichlet concentration. Response weights act only on expert
log scores; categorical counts and Dirichlet terms are never reweighted. Unit
weights recover the ordinary likelihood. Zero response weights are accepted by
the low-level criterion, but the corresponding expert parameters are
unidentified and are therefore not recommended for fitting.

## Release scope

No fitted models, proprietary data, generated paper datasets, predictions,
plots, checkpoints, logs, or manuscript result files are included. This package
contains the model implementation and generic simulation machinery; it is not a
complete reproduction archive for every numerical result in the paper.

Citation details will be added after they are finalized.

