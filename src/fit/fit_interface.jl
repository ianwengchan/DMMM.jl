"""
    fit_LRMoE(Y, X, α_init, model; ...)

Fit an ordinary LRMoE model.

# Arguments
- `Y`: A matrix of response.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `expusure`: an array of numerics, indicating the time invertal over which the count data (if applicable) are collected.
    If `nothing` is provided, it is set to 1.0 for all observations. It is assumed that all continuous expert functions are
    not affected by `exposure`.
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `response_weights`: Fixed nonnegative response powers, one per response
  dimension. `nothing` means all ones. Non-unit weights are supported for
  `exact_Y=true`; they define a weighted composite criterion rather than an
  ordinary likelihood, so AIC and BIC are returned as `NaN`.
- `penalty`: `true` (default) or `false`, indicating whether penalty is imposed on the magnitude of parameters.
- `pen_α`: a numeric penalty on the magnitude of logit regression coefficients. Default is 1.0.
- `pen_params`: an array of penalty term on the magnitude of parameters of component distributions/expert functions.
- `ϵ`: Stopping criterion on loglikelihood (stop when the increment is less than `ϵ`). Default is 0.001.
- `α_iter_max`: Maximum number of iterations when updating `α`. Default is 5.
- `ecm_iter_max`: Maximum number of iterations of the ECM algorithm. Default is 200.
- `grad_jump`: **IN DEVELOPMENT**
- `grad_seq`: **IN DEVELOPMENT**
- `print_steps`: `1` (default) or any integer, indicating whether intermediate updates of parameters should be logged (and how often).
    If `0`, no intermediate updates will be logged. Otherwise, it will be logged every `print_steps` iterations.

# Return Values
- `model_result.α_fit`: Fitted values of logit regression coefficients `α`.
- `model_result.comp_dist`: Fitted parameters of expert functions.
- `converge`: `true` or `false`, indicating whether the fitting procedure has converged.
- `iter`: Number of iterations passed in the fitting function.
- `ll`: Loglikelihood of the fitted model (with penalty on the magnitude of parameters).
- `ll_np`: Loglikelihood of the fitted model (without penalty on the magnitude of parameters).
- `AIC`: Akaike Information Criterion (AIC) of the fitted model.
- `BIC`: Bayesian Information Criterion (BIC) of the fitted model.
"""
function fit_LRMoE(Y, X, α_init, model;
    exposure=nothing,
    response_weights=nothing,
    exact_Y=false,
    penalty=true, pen_α=1.0, pen_params=nothing,
    ϵ=1e-03, α_iter_max=5, ecm_iter_max=200,
    grad_jump=true, grad_seq=nothing,
    zigamma_update_tracker=nothing, iteration_offset=0,
    print_steps=1)

    response_weights = _dmmm_response_weights(response_weights, size(model, 1))
    any(>(0), response_weights) ||
        throw(ArgumentError("at least one response weight must be positive."))
    !exact_Y && _dmmm_is_weighted(response_weights) &&
        throw(ArgumentError("non-unit response_weights currently require exact_Y=true."))
    exact_Y && any(iszero, response_weights) &&
        @warn("Zero-weight response experts are not identified and will remain at their initial values.")

    # Convert possible dataframes to arrays
    # Y = Array(Y)
    # X = Array(X)

    if penalty == false
        pen_α = Inf
        pen_params = [DMMM.no_penalty_init.(model[k, :]) for k in 1:size(model)[1]]
    elseif isnothing(pen_params)
        pen_params = [DMMM.penalty_init.(model[k, :]) for k in 1:size(model)[1]]
    end

    if isnothing(exposure)
        exposure = fill(1.0, size(X)[1])
    end

    if !isinteger(print_steps) || (print_steps < 0)
        error("print_steps must be a nonnegative integer.")
    end

    if exact_Y == true
        tmp = fit_exact(Array(Y), Array(X), Array(α_init), model;
            exposure=exposure,
            response_weights=response_weights,
            penalty=penalty, pen_α=pen_α, pen_params=pen_params,
            ϵ=ϵ, α_iter_max=α_iter_max, ecm_iter_max=ecm_iter_max,
            grad_jump=grad_jump, grad_seq=grad_seq,
            zigamma_update_tracker=zigamma_update_tracker,
            iteration_offset=iteration_offset,
            print_steps=print_steps)
    else
        tmp = fit_main(Array(Y), Array(X), Array(α_init), model;
            exposure=exposure,
            penalty=penalty, pen_α=pen_α, pen_params=pen_params,
            ϵ=ϵ, α_iter_max=α_iter_max, ecm_iter_max=ecm_iter_max,
            grad_jump=grad_jump, grad_seq=grad_seq,
            print_steps=print_steps)
    end

    model_result = LRMoESTD(tmp.α_fit, tmp.model_fit)

    return fit_result = LRMoESTDFit(
        model_result, tmp.converge, tmp.iter, tmp.ll, tmp.ll_np, tmp.AIC, tmp.BIC,
        exact_Y ? tmp.objective_name : :loglik, copy(response_weights),
    )
end

"""
    fit_DMMM(Y, X, group, α_init, ϕ_init, model; ...)

Fit a Dirichlet Mixed-Membership Model to repeated exact observations.
The policyholder-specific membership vector is drawn once,

`P_i | X_i ~ Dirichlet(ϕ * π_i)`,

and is shared by every observation with the same `group` identifier. `X` may
contain one row per observation (constant within policyholder) or one row per
policyholder in first-appearance order. Period-specific exposure belongs in
`exposure`.

# Arguments
- `Y`: An observation-by-response matrix of exact outcomes.
- `X`: Baseline gating covariates.
- `group`: One policyholder identifier per row of `Y`.
- `α_init`: Initial logit-gating coefficients. The last component is the reference.
- `ϕ_init`: Positive initial Dirichlet concentration.
- `model`: Response-by-component matrix of expert distributions.

# Keyword Arguments
- `exact_Y=true`: Only exact responses are currently supported. Passing `false`
  raises an error because the legacy mixture-level truncation normalization is
  not Dirichlet-conjugate.
- `exposure=nothing`: Observation-specific expert exposure; defaults to one.
- `response_weights=nothing`: Fixed nonnegative response powers, one per column
  of `Y`; defaults to all ones. No automatic rescaling. Zero-weight expert rows
  are frozen and not identified by this criterion; at least one weight must be
  positive. Existing penalty terms remain unscaled, so expert updates use
  `response_weights[r] * responsibilities[:, k]` when penalties are enabled.
- `penalty=true`, `pen_α=1.0`, `pen_params=nothing`: Existing LRMoE penalties.
- `estep=:auto`: E-step backend (`:closed_form`, `:enumeration`, `:dp`,
  `:mcem`, or `:auto`). Automatic selection uses the exact closed form for
  panels of at most three periods.
- `ϵ=1e-3`, `ecm_iter_max=200`: Objective convergence controls.
- `dirichlet_iter_max=100`: Maximum iterations in each deterministic `(α, ϕ)` update.
- `min_ϕ=1e-6`, `max_ϕ=1e6`: Robust optimization bounds for the concentration.
- `max_assignments=1_000_000`: Per-policyholder cap on exact latent enumeration.
- `max_dp_states=2_000_000`: Per-policyholder cap on cumulative DP states.
- `rng`, `mcem_burnin`, `mcem_samples`, `mcem_thin`, `mcem_chains`: Collapsed
  Gibbs controls used by MCEM.
- `mcem_callback=nothing`: Optional diagnostic callback after each accepted MCEM
  iteration. Receives copied `alpha`, `phi`, `model`, `iteration`, `q_before`,
  `q_after`, and `conditional_increment`; its return value does not control fitting.
- `mcem_phi_init=:provided`: Use `:profile` to initialize phi from the observed
  criterion with the supplied gates and experts fixed. Response weights are
  respected. Flat profiles retain the supplied value and are flagged.
- `mcem_phi_gate=nothing`: Opt into profile-gated MCEM using an empty NamedTuple
  `(; )` or tolerance overrides. Requires `estep=:mcem`, `update_phi=true`,
  repeated observations, and at most three observations per group. This uses
  exact Dirichlet moments for the scalar criterion, never an exact E-step.
  Defaults use ten-iteration diagnostic windows. Hard convergence requires the
  ordinary MCEM trigger, recent log-phi movement at most
  `phi_stability_tol=0.05`, `gain_tol_per_group=1e-4`, and an implied
  credibility-weight difference at most `credibility_tol=0.01`, using
  `credibility_T=2`. Log-phi profile distance, observed-criterion movement,
  gate TV, and supported-expert changes are diagnostic only. The raw total phi
  score is recorded with a per-group version but is not an acceptance condition.
  Profile intervention defaults to `rescue=:none`. Optional `rescue=:damped`
  requires both `rescue_log_distance=0.75` and
  `rescue_gain_per_group=1e-3` for `rescue_windows=2` scheduled checks, then
  moves `rescue_rho=0.25` of the log-phi distance. It never makes a full jump.
  By default, three replicated E-steps record conditional phi-update Monte
  Carlo variability after the hard criteria pass, but do not block convergence.
  The iteration limit returns `converge=false`.
  With either profile control enabled, callbacks also receive iteration zero
  after initialization. Diagnostics retain initialization, controls, gate events,
  Monte Carlo diagnostic seeds/updates, restart counts and the stopping reason.
- `mcem_expert_update=:coordinate`: Use `:batch` to update the entire expert
  block against one fixed sampled E-step, check the sampled Q once, and use the
  coordinate sampled-Q procedure only if that block check fails.
- `mcem_phi_update=:q`: Use `:ecme` to update phi from the exact conditional
  observed profile after the gate CM step. This is available only for panels
  of at most three periods; it is a special analytical simplification and does
  not claim the same complexity for general panel length.
- `exact_update=:coordinate`: Exact-fitting algorithm. The default preserves
  the coordinate-refresh reference implementation. Use `:block` to update all
  experts from one cached E-step and refresh only at block boundaries.
- `exact_phi_update=:q`: Phi update in exact block mode. Use `:ecme` to maximize
  the integrated criterion after the gate block for panels of at most three
  periods, without an intervening E-step.
- `exact_callback=nothing`: Optional diagnostic callback for exact block fits.
- `update_alpha=true`, `update_phi=true`, `update_experts=true`: Allow selected
  parameter blocks to be held fixed.
- `print_steps=1`: Logging interval; zero disables progress logs.

# Returns
A [`DMMMFit`](@ref) containing the fitted model, period responsibilities,
posterior policyholder membership means, backend metadata, and an objective
trace. Enumeration and DP return ordinary information criteria only for unit
response weights. Non-unit weights record a weighted composite objective and
set AIC/BIC to `NaN`. MCEM leaves
likelihood-based criteria unavailable and records its Monte Carlo `Q` trace.
"""
function fit_DMMM(
    Y,
    X,
    group,
    α_init,
    ϕ_init,
    model;
    exposure=nothing,
    response_weights=nothing,
    exact_Y=true,
    penalty=true,
    pen_α=1.0,
    pen_params=nothing,
    ϵ=1e-3,
    ecm_iter_max=200,
    dirichlet_iter_max=100,
    min_ϕ=1e-6,
    max_ϕ=1e6,
    estep=:auto,
    max_assignments=1_000_000,
    max_dp_states=2_000_000,
    rng=Random.GLOBAL_RNG,
    mcem_burnin=200,
    mcem_samples=500,
    mcem_thin=1,
    mcem_chains=1,
    mcem_min_iterations=5,
    mcem_callback=nothing,
    mcem_phi_init=:provided,
    mcem_phi_gate=nothing,
    mcem_expert_update=:coordinate,
    mcem_phi_update=:q,
    mcem_ecme_stability_window=3,
    mcem_ecme_alpha_tol=1e-3,
    mcem_ecme_log_phi_tol=1e-3,
    mcem_ecme_log_expert_tol=1e-3,
    mcem_ecme_expert_min_support=10.0,
    mcem_observed_trace=false,
    exact_update=:coordinate,
    exact_phi_update=:q,
    exact_callback=nothing,
    zigamma_update_tracker=nothing,
    iteration_offset=0,
    update_alpha=true,
    update_phi=true,
    update_experts=true,
    print_steps=1,
)
    exact_Y isa Bool || throw(ArgumentError("exact_Y must be a Bool value."))
    penalty isa Bool || throw(ArgumentError("penalty must be a Bool value."))
    exact_Y ||
        throw(
            ArgumentError(
                "fit_DMMM currently supports exact responses only. The LRMoE " *
                "mixture-level censoring/truncation normalization is not conjugate to " *
                "the Dirichlet membership distribution."
            ),
        )
    Y_array = Array(Y)
    X_array = Array(X)
    α_array = Array(α_init)
    group_array = collect(group)
    n_observations = size(Y_array, 1)

    n_observations >= 1 || throw(ArgumentError("At least one observation is required."))
    all(isfinite, Y_array) || throw(ArgumentError("Y must contain only finite exact outcomes."))
    all(isfinite, X_array) || throw(ArgumentError("X must contain only finite values."))
    all(isfinite, α_array) || throw(ArgumentError("α_init must contain only finite values."))
    length(group_array) == n_observations ||
        throw(DimensionMismatch("group must contain one identifier per row of Y."))
    size(Y_array, 2) == size(model, 1) ||
        throw(DimensionMismatch("Y and model must have the same number of response dimensions."))
    response_weights = _dmmm_response_weights(response_weights, size(model, 1))
    any(>(0), response_weights) ||
        throw(ArgumentError("At least one response weight must be positive for fitting."))
    any(iszero, response_weights) && update_experts &&
        @warn("Zero-weight response experts are not identified and will remain at their initial values.")
    size(α_array, 1) == size(model, 2) ||
        throw(DimensionMismatch("α and model must have the same number of components."))
    size(α_array, 2) == size(X_array, 2) ||
        throw(DimensionMismatch("α and X must have the same number of covariates."))
    size(model, 2) >= 2 || throw(ArgumentError("DMMM requires at least two components."))
    isfinite(ϕ_init) && ϕ_init > 0 ||
        throw(ArgumentError("ϕ_init must be finite and positive."))
    isfinite(min_ϕ) && isfinite(max_ϕ) && 0 < min_ϕ < max_ϕ ||
        throw(ArgumentError("Require 0 < min_ϕ < max_ϕ < Inf."))
    min_ϕ <= ϕ_init <= max_ϕ ||
        throw(ArgumentError("ϕ_init must lie between min_ϕ and max_ϕ."))
    isfinite(ϵ) && ϵ >= 0 || throw(ArgumentError("ϵ must be finite and nonnegative."))
    ecm_iter_max isa Integer && ecm_iter_max >= 1 ||
        throw(ArgumentError("ecm_iter_max must be a positive integer."))
    dirichlet_iter_max isa Integer && dirichlet_iter_max >= 1 ||
        throw(ArgumentError("dirichlet_iter_max must be a positive integer."))
    max_assignments isa Integer && max_assignments >= 1 ||
        throw(ArgumentError("max_assignments must be a positive integer."))
    max_dp_states isa Integer && max_dp_states >= 1 ||
        throw(ArgumentError("max_dp_states must be a positive integer."))
    rng isa AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG."))
    print_steps isa Integer && print_steps >= 0 ||
        throw(ArgumentError("print_steps must be a nonnegative integer."))
    all(x -> x isa Bool, (update_alpha, update_phi, update_experts)) ||
        throw(ArgumentError("update_alpha, update_phi, and update_experts must be Bool values."))
    exact_update in (:coordinate, :block) || throw(ArgumentError(
        "exact_update must be :coordinate or :block."))
    exact_phi_update in (:q, :ecme) || throw(ArgumentError(
        "exact_phi_update must be :q or :ecme."))
    isfinite(pen_α) && pen_α > 0 ||
        throw(ArgumentError("pen_α must be finite and positive."))

    if penalty == false
        pen_params = [
            DMMM.no_penalty_init.(model[k, :]) for k in 1:size(model, 1)
        ]
    elseif isnothing(pen_params)
        pen_params = [
            DMMM.penalty_init.(model[k, :]) for k in 1:size(model, 1)
        ]
    end
    length(pen_params) == size(model, 1) ||
        throw(DimensionMismatch("pen_params must contain one entry per response dimension."))

    if isnothing(exposure)
        exposure = fill(1.0, n_observations)
    else
        exposure = collect(exposure)
    end
    length(exposure) == n_observations ||
        throw(DimensionMismatch("exposure must contain one value per observation."))
    all(x -> isfinite(x) && x > 0, exposure) ||
        throw(ArgumentError("exposure values must be finite and positive."))

    unused_group_ids, group_index, unused_row_group = _dmmm_groups(group_array)
    resolved_estep = _dmmm_resolve_estep_method(
        estep,
        size(model, 2),
        group_index;
        max_assignments=max_assignments,
        max_dp_states=max_dp_states,
    )
    common_options = (
        exposure=exposure,
        response_weights=response_weights,
        penalty=penalty,
        pen_α=pen_α,
        pen_params=pen_params,
        ϵ=ϵ,
        ecm_iter_max=ecm_iter_max,
        dirichlet_iter_max=dirichlet_iter_max,
        min_ϕ=min_ϕ,
        max_ϕ=max_ϕ,
        update_alpha=update_alpha,
        update_phi=update_phi,
        update_experts=update_experts,
        print_steps=print_steps,
    )
    if resolved_estep == :mcem
        exact_update == :coordinate && exact_phi_update == :q && isnothing(exact_callback) ||
            throw(ArgumentError("Exact update controls require an exact E-step backend."))
        tmp = _dmmm_fit_mcem(
            Y_array,
            X_array,
            group_array,
            α_array,
            ϕ_init,
            model;
            common_options...,
            rng=rng,
            mcem_burnin=mcem_burnin,
            mcem_samples=mcem_samples,
            mcem_thin=mcem_thin,
            mcem_chains=mcem_chains,
            mcem_min_iterations=mcem_min_iterations,
            mcem_callback=mcem_callback,
            mcem_phi_init=mcem_phi_init,
            mcem_phi_gate=mcem_phi_gate,
            mcem_expert_update=mcem_expert_update,
            mcem_phi_update=mcem_phi_update,
            mcem_ecme_stability_window=mcem_ecme_stability_window,
            mcem_ecme_alpha_tol=mcem_ecme_alpha_tol,
            mcem_ecme_log_phi_tol=mcem_ecme_log_phi_tol,
            mcem_ecme_log_expert_tol=mcem_ecme_log_expert_tol,
            mcem_ecme_expert_min_support=mcem_ecme_expert_min_support,
            mcem_observed_trace=mcem_observed_trace,
        )
    else
        isnothing(mcem_callback) || throw(ArgumentError("mcem_callback requires estep=:mcem."))
        mcem_phi_init == :provided && isnothing(mcem_phi_gate) ||
            throw(ArgumentError("MCEM phi initialization and gate require estep=:mcem."))
        mcem_expert_update == :coordinate && mcem_phi_update == :q ||
            throw(ArgumentError("MCEM expert/phi update modes require estep=:mcem."))
        tmp = if exact_update == :coordinate
            isnothing(exact_callback) || throw(ArgumentError(
                "exact_callback currently requires exact_update=:block."))
            exact_phi_update == :q || throw(ArgumentError(
                "The coordinate-refresh reference fitter supports exact_phi_update=:q only."))
            _dmmm_fit_exact(
                Y_array,
                X_array,
                group_array,
                α_array,
                ϕ_init,
                model;
                common_options...,
                estep_method=resolved_estep,
                max_assignments=max_assignments,
                max_dp_states=max_dp_states,
                zigamma_update_tracker=zigamma_update_tracker,
                iteration_offset=iteration_offset,
            )
        else
            _dmmm_fit_exact_block(
            Y_array,
            X_array,
            group_array,
            α_array,
            ϕ_init,
            model;
            common_options...,
            estep_method=resolved_estep,
            max_assignments=max_assignments,
            max_dp_states=max_dp_states,
            exact_phi_update=exact_phi_update,
            exact_callback=exact_callback,
            zigamma_update_tracker=zigamma_update_tracker,
            iteration_offset=iteration_offset,
        )
        end
    end

    model_result = DMMMModel(tmp.α_fit, tmp.ϕ_fit, tmp.model_fit)
    return DMMMFit(
        model_result,
        tmp.converge,
        tmp.iter,
        tmp.ll,
        tmp.ll_np,
        tmp.AIC,
        tmp.BIC,
        tmp.group_ids,
        tmp.responsibilities,
        tmp.posterior_membership,
        tmp.loglik_trace,
        tmp.estep_method,
        tmp.estep_diagnostics,
        tmp.objective_name,
        copy(response_weights),
    )
end
