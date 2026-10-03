_simulate_expert(rng, d::BurrExpert) = rand(rng, Burr(d.k, d.c, d.λ))
_simulate_expert(rng, d::GammaExpert) = rand(rng, Gamma(d.k, d.θ))
_simulate_expert(rng, d::InverseGaussianExpert) = rand(rng, InverseGaussian(d.μ, d.λ))
_simulate_expert(rng, d::LogNormalExpert) = rand(rng, LogNormal(d.μ, d.σ))
_simulate_expert(rng, d::NormalExpert) = rand(rng, Normal(d.μ, d.σ))
_simulate_expert(rng, d::WeibullExpert) = rand(rng, Weibull(d.k, d.θ))
_simulate_expert(rng, d::BinomialExpert) = rand(rng, Binomial(d.n, d.p))
_simulate_expert(rng, d::GammaCountExpert) = rand(rng, GammaCount(d.m, d.s))
_simulate_expert(rng, d::NegativeBinomialExpert) = rand(rng, Distributions.NegativeBinomial(d.n, d.p))
_simulate_expert(rng, d::PoissonExpert) = rand(rng, Poisson(d.λ))

function _simulate_expert(rng, d::ZOIBetaExpert)
    u = rand(rng)
    u <= d.p0 && return 0.0
    u >= 1 - d.p1 && return 1.0
    return rand(rng, Distributions.Beta(d.α, d.β))
end

_simulate_expert(rng, d::ZIBurrExpert) =
    rand(rng) < d.p ? 0.0 : rand(rng, Burr(d.k, d.c, d.λ))
_simulate_expert(rng, d::ZIGammaExpert) =
    rand(rng) < d.p ? 0.0 : rand(rng, Gamma(d.k, d.θ))
_simulate_expert(rng, d::ZIInverseGaussianExpert) =
    rand(rng) < d.p ? 0.0 : rand(rng, InverseGaussian(d.μ, d.λ))
_simulate_expert(rng, d::ZILogNormalExpert) =
    rand(rng) < d.p ? 0.0 : rand(rng, LogNormal(d.μ, d.σ))
_simulate_expert(rng, d::ZIWeibullExpert) =
    rand(rng) < d.p ? 0.0 : rand(rng, Weibull(d.k, d.θ))
_simulate_expert(rng, d::ZIBinomialExpert) =
    rand(rng) < d.p0 ? 0 : rand(rng, Binomial(d.n, d.p))
_simulate_expert(rng, d::ZIGammaCountExpert) =
    rand(rng) < d.p ? 0 : rand(rng, GammaCount(d.m, d.s))
_simulate_expert(rng, d::ZINegativeBinomialExpert) =
    rand(rng) < d.p0 ? 0 : rand(rng, Distributions.NegativeBinomial(d.n, d.p))
_simulate_expert(rng, d::ZIPoissonExpert) =
    rand(rng) < d.p ? 0 : rand(rng, Poisson(d.λ))

sim_components(rng, model) = _simulate_expert.(Ref(rng), model)

function sim_logit_gating(rng, α, X)
    X = Array(X)
    probs = exp.(LogitGating(α, X))
    return hcat([rand(rng, Distributions.Multinomial(1, probs[i, :])) for i in 1:size(X)[1]]...)'
end

function sim_dataset(α, X, model; exposure=nothing, rng=Random.GLOBAL_RNG)
    X = Array(X)
    if isnothing(exposure)
        exposure = fill(1.0, size(X)[1])
    end
    model_expo = exposurize_model(model; exposure=exposure)
    gating_sim = sim_logit_gating(rng, α, X)
    return vcat(
        [sim_components(rng, model_expo[:, :, i]) * gating_sim[i, :] for i in 1:size(X)[1]]'...
    )
end

"""
    simulate_dmmm(α, ϕ, X, group, model; exposure=nothing,
                  rng=Random.GLOBAL_RNG, return_latent=false)

Simulate a DMMM panel. One fixed membership vector is drawn for each
policyholder and reused for every matching `group` identifier. `X` may have one
row per observation (constant within policyholder) or one row per policyholder.
Pass a seeded `AbstractRNG` for reproducible simulation.
"""
function simulate_dmmm(
    α,
    ϕ,
    X,
    group,
    model;
    exposure=nothing,
    rng=Random.GLOBAL_RNG,
    return_latent=false,
)
    return_latent isa Bool ||
        throw(ArgumentError("return_latent must be a Bool value."))
    group_array = collect(group)
    n_observations = length(group_array)
    n_observations >= 1 || throw(ArgumentError("At least one observation is required."))
    group_ids, group_index, row_group = _dmmm_groups(group_array)
    X_group = _dmmm_group_covariates(Array(X), group_index, n_observations)
    α_ref = _dmmm_canonical_alpha(α)
    size(α_ref, 1) == size(model, 2) ||
        throw(DimensionMismatch("α and model must have the same number of components."))
    size(α_ref, 2) == size(X_group, 2) ||
        throw(DimensionMismatch("α and X must have the same number of covariates."))
    isfinite(ϕ) && ϕ > 0 ||
        throw(ArgumentError("The Dirichlet concentration ϕ must be finite and positive."))

    if isnothing(exposure)
        exposure = fill(1.0, n_observations)
    else
        exposure = collect(exposure)
    end
    length(exposure) == n_observations ||
        throw(DimensionMismatch("exposure must contain one value per observation."))
    all(x -> isfinite(x) && x > 0, exposure) ||
        throw(ArgumentError("exposure values must be finite and positive."))

    gate_prob = _dmmm_gate_probabilities(α_ref, X_group)
    membership = Matrix{Float64}(undef, length(group_ids), size(model, 2))
    for i in eachindex(group_ids)
        membership[i, :] =
            rand(rng, Distributions.Dirichlet(ϕ .* vec(gate_prob[i, :])))
    end

    model_exposure = exposurize_model(model; exposure=exposure)
    response = Matrix{Float64}(undef, n_observations, size(model, 1))
    component = Vector{Int}(undef, n_observations)
    for row in 1:n_observations
        component[row] =
            rand(rng, Distributions.Categorical(vec(membership[row_group[row], :])))
        simulated = sim_components(rng, model_exposure[:, :, row])
        response[row, :] = simulated[:, component[row]]
    end

    return if return_latent
        (
            Y=response,
            component=component,
            membership=membership,
            group_ids=group_ids,
        )
    else
        response
    end
end
