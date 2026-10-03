"""
    predict_class_prior(X, α)

Predicts the latent class probabilities, 
given covariates `X` 
and logit regression coefficients `α`.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.

# Return Values
- `prob`: A matrix of latent class probabilities.
- `max_prob_idx`: A matrix of the most likely latent class for each observation.
"""
function predict_class_prior(X, α)
    tmp = exp.(LogitGating(α, X))
    return (prob=tmp, max_prob_idx=[findmax(tmp[i, :])[2] for i in 1:size(tmp)[1]])
end

"""
    predict_class_posterior(Y, X, α, model; 
        exact_Y = true, exposure_past = nothing)

Predicts the latent class probabilities, 
given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions. 

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- `prob`: A matrix of latent class probabilities.
- `max_prob_idx`: A matrix of the most likely latent class for each observation.
"""
function predict_class_posterior(Y, X, α, model; exact_Y=true, exposure_past=nothing)
    if exact_Y == true
        Y = _exact_to_full(Y)
    end

    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_past)

    gate = LogitGating(α, X)
    ll_np_list = loglik_np(Y, gate, model_exp)
    z_e_obs = EM_E_z_obs(ll_np_list.gate_expert_ll_comp, ll_np_list.gate_expert_ll)
    return (
        prob=z_e_obs, max_prob_idx=[findmax(z_e_obs[i, :])[2] for i in 1:size(z_e_obs)[1]]
    )
end

"""
    predict_class_posterior_dmmm(Y, X, group, α, ϕ, model;
                                    exposure_past=nothing)

Compute period responsibilities and posterior mean membership vectors
for a fitted Dirichlet Mixed-Membership Model.
`response_weights` applies fixed powers to the history likelihood only. With
non-unit weights these are generalized posterior probabilities. Passing a fit
object inherits its stored weights; explicitly pass `response_weights=nothing`
to use the ordinary history likelihood instead.
"""
function predict_class_posterior_dmmm(
    Y,
    X,
    group,
    α,
    ϕ,
    model;
    exposure_past=nothing,
    response_weights=nothing,
    estep=:auto,
    max_assignments=1_000_000,
    max_dp_states=2_000_000,
    rng=Random.GLOBAL_RNG,
    mcem_burnin=200,
    mcem_samples=500,
    mcem_thin=1,
    mcem_chains=1,
)
    Y_array = Array(Y)
    size(Y_array, 2) == size(model, 1) ||
        throw(DimensionMismatch("Y and model must have the same number of response dimensions."))
    all(isfinite, Y_array) ||
        throw(ArgumentError("Y must contain only finite exact outcomes."))
    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(Y_array, 1))
    else
        exposure_past = collect(exposure_past)
    end
    length(exposure_past) == size(Y_array, 1) ||
        throw(DimensionMismatch("exposure_past must contain one value per observation."))
    all(x -> isfinite(x) && x > 0, exposure_past) ||
        throw(ArgumentError("exposure_past values must be finite and positive."))

    response_weights = _dmmm_response_weights(response_weights, size(model, 1))
    expert_loglik = _dmmm_expert_logscores(
        Y_array, model, exposure_past, response_weights
    )
    result = dmmm_estep(
        expert_loglik,
        Array(X),
        collect(group),
        α,
        ϕ;
        method=estep,
        max_assignments=max_assignments,
        max_dp_states=max_dp_states,
        rng=rng,
        mcem_burnin=mcem_burnin,
        mcem_samples=mcem_samples,
        mcem_thin=mcem_thin,
        mcem_chains=mcem_chains,
    )
    return (
        prob=result.posterior_membership,
        max_prob_idx=[
            findmax(result.posterior_membership[i, :])[2] for
            i in axes(result.posterior_membership, 1)
        ],
        responsibilities=result.responsibilities,
        posterior_membership=result.posterior_membership,
        prior_membership=result.gate_prob,
        experience_membership=result.experience_membership,
        credibility=result.credibility,
        group_ids=result.group_ids,
        estep_method=result.method,
        estep_diagnostics=result.diagnostics,
        response_weights=copy(response_weights),
        posterior_kind=_dmmm_is_weighted(response_weights) ? :generalized : :ordinary,
    )
end

"""
    predict_var_posterior_dmmm(Y, X, group, α, ϕ, model;
                                  exposure_past=nothing,
                                  exposure_future=nothing)

Predict the marginal variance of each coordinate of one future response per
policyholder.
"""
function predict_var_posterior_dmmm(
    Y,
    X,
    group,
    α,
    ϕ,
    model;
    exposure_past=nothing,
    exposure_future=nothing,
    response_weights=nothing,
    estep=:auto,
    max_assignments=1_000_000,
    max_dp_states=2_000_000,
    rng=Random.GLOBAL_RNG,
    mcem_burnin=200,
    mcem_samples=500,
    mcem_thin=1,
    mcem_chains=1,
)
    posterior = predict_class_posterior_dmmm(
        Y,
        X,
        group,
        α,
        ϕ,
        model;
        exposure_past=exposure_past,
        response_weights=response_weights,
        estep=estep,
        max_assignments=max_assignments,
        max_dp_states=max_dp_states,
        rng=rng,
        mcem_burnin=mcem_burnin,
        mcem_samples=mcem_samples,
        mcem_thin=mcem_thin,
        mcem_chains=mcem_chains,
    )
    n_groups = length(posterior.group_ids)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, n_groups)
    else
        exposure_future = collect(exposure_future)
    end
    length(exposure_future) == n_groups ||
        throw(DimensionMismatch("exposure_future must contain one value per policyholder."))
    all(x -> isfinite(x) && x > 0, exposure_future) ||
        throw(ArgumentError("exposure_future values must be finite and positive."))

    model_exposure = exposurize_model(model; exposure=exposure_future)
    component_mean = mean.(model_exposure)
    component_var = var.(model_exposure)
    result = Matrix{Float64}(undef, n_groups, size(model, 1))
    for i in 1:n_groups
        weights = vec(posterior.posterior_membership[i, :])
        predictive_mean = component_mean[:, :, i] * weights
        result[i, :] =
            component_var[:, :, i] * weights +
            (
                component_mean[:, :, i] .-
                reshape(predictive_mean, length(predictive_mean), 1)
            ) .^ 2 * weights
    end
    return result
end

function predict_var_posterior_dmmm(
    Y,
    X,
    group,
    fit::DMMMFit;
    response_weights=fit.response_weights,
    kwargs...,
)
    return predict_var_posterior_dmmm(
        Y,
        X,
        group,
        fit.model_fit.α,
        fit.model_fit.ϕ,
        fit.model_fit.comp_dist;
        response_weights=response_weights,
        kwargs...,
    )
end

function predict_class_posterior_dmmm(
    Y,
    X,
    group,
    fit::DMMMFit;
    response_weights=fit.response_weights,
    kwargs...,
)
    return predict_class_posterior_dmmm(
        Y,
        X,
        group,
        fit.model_fit.α,
        fit.model_fit.ϕ,
        fit.model_fit.comp_dist;
        response_weights=response_weights,
        kwargs...,
    )
end

"""
    DMMMPredictiveMixture(weights, experts)

A finite one-dimensional predictive mixture. `weights` are class probabilities
and `experts` are the corresponding exposure-adjusted expert distributions.
Use the usual `pdf`, `logpdf`, `cdf`, `mean`, `var`, and `rand` functions.
"""
struct DMMMPredictiveMixture
    weights::Vector{Float64}
    experts::Vector{Any}
    function DMMMPredictiveMixture(weights, experts)
        length(weights) == length(experts) ||
            throw(DimensionMismatch("weights and experts must have equal length."))
        isempty(weights) && throw(ArgumentError("A predictive mixture cannot be empty."))
        values = Float64.(weights)
        all(x -> isfinite(x) && x >= 0, values) ||
            throw(ArgumentError("Mixture weights must be finite and nonnegative."))
        total = sum(values)
        total > 0 || throw(ArgumentError("At least one mixture weight must be positive."))
        values ./= total
        return new(values, Any[experts...])
    end
end

_full_expert_cdf(d, x) = cdf(d, x)
_full_expert_cdf(d::ZIBurrExpert, x) =
    x < 0 ? 0.0 : d.p + (1 - d.p) * cdf(Burr(d.k, d.c, d.λ), x)
_full_expert_cdf(d::ZIGammaExpert, x) =
    x < 0 ? 0.0 : d.p + (1 - d.p) * cdf(Gamma(d.k, d.θ), x)
_full_expert_cdf(d::ZIInverseGaussianExpert, x) =
    x < 0 ? 0.0 : d.p + (1 - d.p) * cdf(InverseGaussian(d.μ, d.λ), x)
_full_expert_cdf(d::ZILogNormalExpert, x) =
    x < 0 ? 0.0 : d.p + (1 - d.p) * cdf(LogNormal(d.μ, d.σ), x)
_full_expert_cdf(d::ZIWeibullExpert, x) =
    x < 0 ? 0.0 : d.p + (1 - d.p) * cdf(Weibull(d.k, d.θ), x)
_full_expert_cdf(d::ZIBinomialExpert, x) =
    x < 0 ? 0.0 : d.p0 + (1 - d.p0) * cdf(Binomial(d.n, d.p), x)
_full_expert_cdf(d::ZIGammaCountExpert, x) =
    x < 0 ? 0.0 : d.p + (1 - d.p) * cdf(GammaCount(d.m, d.s), x)
_full_expert_cdf(d::ZINegativeBinomialExpert, x) =
    x < 0 ? 0.0 : d.p0 + (1 - d.p0) * cdf(Distributions.NegativeBinomial(d.n, d.p), x)
_full_expert_cdf(d::ZIPoissonExpert, x) =
    x < 0 ? 0.0 : d.p + (1 - d.p) * cdf(Poisson(d.λ), x)
function _full_expert_cdf(d::ZOIBetaExpert, x)
    x < 0 && return 0.0
    x >= 1 && return 1.0
    return d.p0 + (1 - d.p0 - d.p1) * cdf(Distributions.Beta(d.α, d.β), x)
end

pdf(d::DMMMPredictiveMixture, x::Real) =
    sum(d.weights .* exp.(expert_ll_exact.(d.experts, x)))
logpdf(d::DMMMPredictiveMixture, x::Real) =
    logsumexp(log.(d.weights) .+ expert_ll_exact.(d.experts, x))
cdf(d::DMMMPredictiveMixture, x::Real) =
    sum(d.weights .* _full_expert_cdf.(d.experts, x))
mean(d::DMMMPredictiveMixture) = sum(d.weights .* mean.(d.experts))
function var(d::DMMMPredictiveMixture)
    component_means = mean.(d.experts)
    mixture_mean = sum(d.weights .* component_means)
    return sum(d.weights .* (var.(d.experts) .+ (component_means .- mixture_mean) .^ 2))
end
rand(rng::AbstractRNG, d::DMMMPredictiveMixture) =
    _simulate_expert(rng, d.experts[rand(rng, Categorical(d.weights))])
rand(d::DMMMPredictiveMixture) = rand(Random.GLOBAL_RNG, d)

"""
    predictive_mixture_dmmm(Y, X, group, alpha, phi, model; ...)

Return a matrix with one finite predictive mixture per policyholder and response
dimension. History updates use the same exact or approximate E-step controls as
`predict_class_posterior_dmmm`; experts are adjusted to `exposure_future`.
"""
function predictive_mixture_dmmm(
    Y,
    X,
    group,
    alpha,
    phi,
    model;
    exposure_past=nothing,
    exposure_future=nothing,
    response_weights=nothing,
    kwargs...,
)
    posterior = predict_class_posterior_dmmm(
        Y,
        X,
        group,
        alpha,
        phi,
        model;
        exposure_past=exposure_past,
        response_weights=response_weights,
        kwargs...,
    )
    n_groups = length(posterior.group_ids)
    future = isnothing(exposure_future) ? fill(1.0, n_groups) : collect(exposure_future)
    length(future) == n_groups ||
        throw(DimensionMismatch("exposure_future must contain one value per policyholder."))
    all(x -> isfinite(x) && x > 0, future) ||
        throw(ArgumentError("exposure_future values must be finite and positive."))
    adjusted = exposurize_model(model; exposure=future)
    mixtures = Matrix{DMMMPredictiveMixture}(undef, n_groups, size(model, 1))
    for i in 1:n_groups, response in axes(model, 1)
        mixtures[i, response] = DMMMPredictiveMixture(
            vec(posterior.posterior_membership[i, :]),
            [adjusted[response, component, i] for component in axes(model, 2)],
        )
    end
    return mixtures
end

function predictive_mixture_dmmm(
    Y,
    X,
    group,
    fit::DMMMFit;
    response_weights=fit.response_weights,
    kwargs...,
)
    return predictive_mixture_dmmm(
        Y,
        X,
        group,
        fit.model_fit.α,
        fit.model_fit.ϕ,
        fit.model_fit.comp_dist;
        response_weights=response_weights,
        kwargs...,
    )
end

"""
    predict_mean_prior(X, α, model; 
        exposure_future = nothing)

Predicts the mean values of response, 
given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted mean values of response, based on prior probabilities.
"""
function predict_mean_prior(X, α, model; exposure_future=nothing)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights = predict_class_prior(X, α).prob
    means = mean.(model_exp)

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        result[i, :] = means[:, :, i] * weights[i, :]
    end

    return result
end

"""
    predict_mean_posterior(Y, X, α, model; 
        exact_Y = true, exposure_past = nothing, exposure_future = nothing)

Predicts the mean values of response,
given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted mean values of response, based on posterior probabilities.
"""
function predict_mean_posterior(
    Y, X, α, model; exact_Y=true, exposure_past=nothing, exposure_future=nothing
)
    # if exact_Y == true
    #     Y = _exact_to_full(Y)
    # end

    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights =
        predict_class_posterior(
            Y, X, α, model; exact_Y=exact_Y, exposure_past=exposure_past
        ).prob
    means = mean.(model_exp)

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        result[i, :] = means[:, :, i] * weights[i, :]
    end

    return result
end

"""
    predict_mean_posterior_dmmm(Y, X, group, α, ϕ, model;
                                   exposure_past=nothing,
                                   exposure_future=nothing)

Predict one future response per policyholder. Expert means are evaluated at
the future-period exposure and averaged using the posterior mean membership
vector.
"""
function predict_mean_posterior_dmmm(
    Y,
    X,
    group,
    α,
    ϕ,
    model;
    exposure_past=nothing,
    exposure_future=nothing,
    response_weights=nothing,
    estep=:auto,
    max_assignments=1_000_000,
    max_dp_states=2_000_000,
    rng=Random.GLOBAL_RNG,
    mcem_burnin=200,
    mcem_samples=500,
    mcem_thin=1,
    mcem_chains=1,
)
    posterior = predict_class_posterior_dmmm(
        Y,
        X,
        group,
        α,
        ϕ,
        model;
        exposure_past=exposure_past,
        response_weights=response_weights,
        estep=estep,
        max_assignments=max_assignments,
        max_dp_states=max_dp_states,
        rng=rng,
        mcem_burnin=mcem_burnin,
        mcem_samples=mcem_samples,
        mcem_thin=mcem_thin,
        mcem_chains=mcem_chains,
    )
    n_groups = length(posterior.group_ids)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, n_groups)
    else
        exposure_future = collect(exposure_future)
    end
    length(exposure_future) == n_groups ||
        throw(DimensionMismatch("exposure_future must contain one value per policyholder."))
    all(x -> isfinite(x) && x > 0, exposure_future) ||
        throw(ArgumentError("exposure_future values must be finite and positive."))

    model_exposure = exposurize_model(model; exposure=exposure_future)
    component_mean = mean.(model_exposure)
    result = Matrix{Float64}(undef, n_groups, size(model, 1))
    for i in 1:n_groups
        result[i, :] =
            component_mean[:, :, i] * vec(posterior.posterior_membership[i, :])
    end
    return result
end

function predict_mean_posterior_dmmm(
    Y,
    X,
    group,
    fit::DMMMFit;
    response_weights=fit.response_weights,
    kwargs...,
)
    return predict_mean_posterior_dmmm(
        Y,
        X,
        group,
        fit.model_fit.α,
        fit.model_fit.ϕ,
        fit.model_fit.comp_dist;
        response_weights=response_weights,
        kwargs...,
    )
end

"""
    predict_var_prior(X, α, model; 
        exposure_future = nothing)

Predicts the variance of response, 
given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `exposure_future`: A vector indicating the time exposure of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted variance of response, based on prior probabilities.
"""
function predict_var_prior(X, α, model; exposure_future=nothing)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights = predict_class_prior(X, α).prob

    c_mean = mean.(model_exp)
    g_mean = predict_mean_prior(X, α, model; exposure_future=exposure_future)
    c_var = var.(model_exp)

    var_c_mean = fill(NaN, size(X)[1], size(model)[1])
    mean_c_var = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        var_c_mean[i, :] = (c_mean[:, :, i] .- g_mean[i, :]) .^ 2 * weights[i, :]
        mean_c_var[i, :] = c_var[:, :, i] * weights[i, :]
    end

    return var_c_mean + mean_c_var
end

"""
    predict_var_posterior(Y, X, α, model; 
        exact_Y = true, exposure_past = nothing, exposure_future = nothing)

Predicts the variance of response, 
given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted variance of response, based on posterior probabilities.
"""
function predict_var_posterior(
    Y, X, α, model; exact_Y=true, exposure_past=nothing, exposure_future=nothing
)
    # if exact_Y == true
    #     Y = _exact_to_full(Y)
    # end

    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights =
        predict_class_posterior(
            Y, X, α, model; exact_Y=exact_Y, exposure_past=exposure_past
        ).prob

    c_mean = mean.(model_exp)
    g_mean = predict_mean_posterior(
        Y,
        X,
        α,
        model;
        exact_Y=exact_Y,
        exposure_past=exposure_past,
        exposure_future=exposure_future,
    )
    c_var = var.(model_exp)

    var_c_mean = fill(NaN, size(X)[1], size(model)[1])
    mean_c_var = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        var_c_mean[i, :] = (c_mean[:, :, i] .- g_mean[i, :]) .^ 2 * weights[i, :]
        mean_c_var[i, :] = c_var[:, :, i] * weights[i, :]
    end

    return var_c_mean + mean_c_var
end

"""
    predict_limit_prior(X, α, model, limit; 
        exposure_future = nothing)

Predicts the limit expected value (LEV) of response, 
given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `limit`: A matrix specifying the cutoff point.

# Optional Arguments
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted limit expected value of response, based on prior probabilities.
"""
function predict_limit_prior(X, α, model, limit; exposure_future=nothing)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights = predict_class_prior(X, α).prob

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        means = vcat(
            [
                hcat(
                    [lev(model_exp[d, j, i], limit[i, d]) for j in 1:size(model_exp)[2]]...
                ) for d in 1:size(model_exp)[1]
            ]...,
        )
        result[i, :] = means * weights[i, :]
    end

    return result
end

"""
    predict_limit_posterior(Y, X, α, model, limit;
        exact_Y = true, exposure_past = nothing, exposure_future = nothing)

Predicts the limit expected value (LEV) of response, 
given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `limit`: A vector specifying the cutoff point.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted limit expected value of response, based on posterior probabilities.
"""
function predict_limit_posterior(
    Y, X, α, model, limit; exact_Y=true, exposure_past=nothing, exposure_future=nothing
)
    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights =
        predict_class_posterior(
            Y, X, α, model; exact_Y=exact_Y, exposure_past=exposure_past
        ).prob

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        means = vcat(
            [
                hcat(
                    [lev(model_exp[d, j, i], limit[i, d]) for j in 1:size(model_exp)[2]]...
                ) for d in 1:size(model_exp)[1]
            ]...,
        )
        result[i, :] = means * weights[i, :]
    end

    return result
end

"""
    predict_excess_prior(X, α, model, limit;
        exposure_future = nothing)

Predicts the excess expectation of response, 
given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `limit`: A vector specifying the cutoff point.

# Optional Arguments
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted excess expectation of response, based on prior probabilities.
"""
function predict_excess_prior(X, α, model, limit; exposure_future=nothing)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights = predict_class_prior(X, α).prob

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        means = vcat(
            [
                hcat(
                    [
                        excess(model_exp[d, j, i], limit[i, d]) for
                        j in 1:size(model_exp)[2]
                    ]...,
                ) for d in 1:size(model_exp)[1]
            ]...,
        )
        result[i, :] = means * weights[i, :]
    end

    return result
end

"""
    predict_excess_posterior(Y, X, α, model, limit;
        exact_Y = true, exposure_past = nothing, exposure_future = nothing)

Predicts the excess expectation of response, 
given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `limit`: A vector specifying the cutoff point.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted excess expectation of response, based on posterior probabilities.
"""
function predict_excess_posterior(
    Y, X, α, model, limit; exact_Y=true, exposure_past=nothing, exposure_future=nothing
)
    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights =
        predict_class_posterior(
            Y, X, α, model; exact_Y=exact_Y, exposure_past=exposure_past
        ).prob

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        means = vcat(
            [
                hcat(
                    [
                        excess(model_exp[d, j, i], limit[i, d]) for
                        j in 1:size(model_exp)[2]
                    ]...,
                ) for d in 1:size(model_exp)[1]
            ]...,
        )
        result[i, :] = means * weights[i, :]
    end

    return result
end

# solve a quantile of a mixture model
# Bisection method seems to give the most stable results
function _solve_continuous_mix_quantile(weights, experts, p)
    p0 = sum(weights .* exp.(expert_ll.(experts, 0.0, 0.0, 0.0, Inf)))
    if p <= p0
        return 0.0
    else
        # init_guess = minimum([maximum(quantile.(experts, 0.90)) 500])
        # init_guess = maximum([maximum(quantile.(experts, 0.90)) 1000])
        init_guess = maximum([maximum(quantile.(experts, p)) 1000])
        VaR = try
            # Roots.find_zero(y -> sum(weights .* exp.(expert_ll.(experts, 0.0, 0.0, y, Inf))) - p, init_guess, Roots.Order2())
            Roots.find_zero(
                y -> sum(weights .* exp.(expert_ll.(experts, 0.0, 0.0, y, Inf))) - p,
                (0.0, init_guess + 500),
            )
        catch
            NaN
        end
        return VaR
    end
end

function _calc_continuous_CTE(weights, experts, p, VaR)
    m = sum(vec(weights) .* vec(mean.(experts)))
    lim_ev = sum(vec(weights) .* vec(lev.(experts, VaR))) # [lev(model[k], VaR) for k in 1:length(model)]
    return VaR + (m - lim_ev) / (1 - p)
end

# calculate CTE based on VaR and p
# function _calc_continuous_mix_CTE(weights, experts, p, means, VaR)
#     return 
# end

"""
    predict_VaRCTE_prior(X, α, model, p;
        exposure_future = nothing)

Predicts the `p`-th value-at-risk (VaR) and conditional tail expectation (CTE) of response, 
given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `p`: A matrix of probabilities.

# Optional Arguments
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- `VaR`: A matrix of predicted VaR of response, based on prior probabilities.
- `CTE`: A matrix of predicted CTE of response, based on prior probabilities.
"""
function predict_VaRCTE_prior(X, α, model, p; exposure_future=nothing)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights = predict_class_prior(X, α).prob

    VaR = fill(NaN, size(X)[1], size(model)[1])
    CTE = fill(NaN, size(X)[1], size(model)[1])
    for i in 1:size(X)[1]
        for k in 1:size(model)[1]
            VaR[i, k] = _solve_continuous_mix_quantile(
                weights[i, :], model_exp[k, :, i], p[i, k]
            )
            CTE[i, k] = _calc_continuous_CTE(
                weights[i, :], model_exp[k, :, i], p[i, k], VaR[i, k]
            )
        end
    end
    # return VaR

    # VaR = vcat([hcat([_solve_continuous_mix_quantile(weights[i,:], model[k,:], p) for k in 1:size(model)[1] ]...) for i in 1:size(X)[1]]...)
    # CTE = vcat([hcat([_calc_continuous_CTE(weights[i,:], model[k,:], p, VaR[i,k]) for k in 1:size(model)[1] ]...) for i in 1:size(X)[1]]...)
    return (VaR=VaR, CTE=CTE)
end

"""
    predict_VaRCTE_posterior(Y, X, α, model, p;
        exact_Y = true, exposure_past = nothing, exposure_future = nothing)

Predicts the `p`-th value-at-risk (VaR) and conditional tail expectation (CTE) of response, 
given observations `Y`, covariates `X`,
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `p`: A matrix of probabilities.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- `VaR`: A matrix of predicted VaR of response, based on posterior probabilities.
- `CTE`: A matrix of predicted CTE of response, based on posterior probabilities.
"""
function predict_VaRCTE_posterior(
    Y, X, α, model, p; exact_Y=true, exposure_past=nothing, exposure_future=nothing
)
    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights =
        predict_class_posterior(
            Y, X, α, model; exact_Y=exact_Y, exposure_past=exposure_past
        ).prob

    VaR = fill(NaN, size(X)[1], size(model)[1])
    CTE = fill(NaN, size(X)[1], size(model)[1])
    for i in 1:size(X)[1]
        for k in 1:size(model)[1]
            VaR[i, k] = _solve_continuous_mix_quantile(
                weights[i, :], model_exp[k, :, i], p[i, k]
            )
            CTE[i, k] = _calc_continuous_CTE(
                weights[i, :], model_exp[k, :, i], p[i, k], VaR[i, k]
            )
        end
    end
    # return VaR

    # VaR = vcat([hcat([_solve_continuous_mix_quantile(weights[i,:], model[k,:], p) for k in 1:size(model)[1] ]...) for i in 1:size(X)[1]]...)
    # CTE = vcat([hcat([_calc_continuous_CTE(weights[i,:], model[k,:], p, VaR[i,k]) for k in 1:size(model)[1] ]...) for i in 1:size(X)[1]]...)
    return (VaR=VaR, CTE=CTE)
end
