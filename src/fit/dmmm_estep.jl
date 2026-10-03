const _DMMM_ESTEP_METHODS = (:auto, :closed_form, :enumeration, :dp, :mcem)

function _dmmm_logaddexp(a, b)
    a == -Inf && return b
    b == -Inf && return a
    larger = max(a, b)
    return larger + log1p(exp(min(a, b) - larger))
end

function _dmmm_increment_count(count::Tuple, component)
    return ntuple(k -> count[k] + (k == component), length(count))
end

function _dmmm_count_state_total(n_components, n_periods)
    # Sum_{t=0}^T binomial(t + K - 1, K - 1) = binomial(T + K, K).
    return binomial(big(n_periods + n_components), big(n_components))
end

function _dmmm_assignment_count(n_components, n_periods)
    return big(n_components)^n_periods
end

function _dmmm_resolve_estep_method(
    method,
    n_components,
    group_index;
    max_assignments,
    max_dp_states,
)
    method isa Symbol && method in _DMMM_ESTEP_METHODS ||
        throw(
            ArgumentError(
                "method must be one of :auto, :closed_form, :enumeration, :dp, or :mcem."
            ),
        )
    method != :auto && return method

    max_periods = maximum(length, group_index)
    if max_periods <= 3
        return :closed_form
    elseif _dmmm_assignment_count(n_components, max_periods) <= max_assignments
        return :enumeration
    elseif _dmmm_count_state_total(n_components, max_periods) <= max_dp_states
        return :dp
    end
    return :mcem
end

include("dmmm_stable_statistics.jl")

function _dmmm_validate_estep_inputs(
    expert_loglik,
    X,
    group,
    α,
    ϕ,
)
    n_observations, n_components = size(expert_loglik)
    n_observations >= 1 || throw(ArgumentError("At least one observation is required."))
    n_components >= 2 || throw(ArgumentError("DMMM requires at least two components."))
    length(group) == n_observations ||
        throw(DimensionMismatch("group must contain one identifier per observation."))
    size(α, 1) == n_components ||
        throw(DimensionMismatch("α and expert_loglik must have the same number of components."))
    isfinite(ϕ) && ϕ > 0 ||
        throw(ArgumentError("The Dirichlet concentration ϕ must be finite and positive."))
    all(isfinite, X) || throw(ArgumentError("X must contain only finite values."))
    all(isfinite, α) || throw(ArgumentError("α must contain only finite values."))
    any(isnan, expert_loglik) &&
        throw(ArgumentError("expert_loglik cannot contain NaN values."))
    any(x -> x == Inf, expert_loglik) &&
        throw(ArgumentError("expert_loglik cannot contain positive Inf values."))

    group_ids, group_index, row_group = _dmmm_groups(group)
    X_group = _dmmm_group_covariates(X, group_index, n_observations)
    size(α, 2) == size(X_group, 2) ||
        throw(DimensionMismatch("The number of columns of α and X must agree."))
    gate_prob = _dmmm_gate_probabilities(_dmmm_canonical_alpha(α), X_group)
    return (
        n_observations=n_observations,
        n_components=n_components,
        group_ids=group_ids,
        group_index=group_index,
        row_group=row_group,
        X_group=X_group,
        gate_prob=gate_prob,
    )
end

function _dmmm_finish_estep(
    method,
    diagnostics,
    ll,
    group_loglik,
    responsibilities,
    expected_log_membership,
    validated,
    ϕ;
    log_responsibilities=log.(responsibilities),
)
    n_groups = length(validated.group_ids)
    posterior_membership = zeros(Float64, n_groups, validated.n_components)
    experience_membership = zeros(Float64, n_groups, validated.n_components)
    credibility = zeros(Float64, n_groups)
    ranges = _dmmm_policy_ranges(n_groups)
    @threads :static for chunk in eachindex(ranges)
        for i in ranges[chunk]
            rows = validated.group_index[i]
            n_periods = length(rows)
            expected_counts = [exp(_dmmm_logweight_sum(view(log_responsibilities,:,k);rows=rows))
                for k in 1:validated.n_components]
            experience_membership[i, :] = expected_counts ./ n_periods
            credibility[i] = n_periods / (ϕ + n_periods)
            posterior_membership[i, :] =
                (ϕ .* vec(validated.gate_prob[i, :]) .+ expected_counts) ./
                (ϕ + n_periods)
        end
    end
    return (
        method=method,
        diagnostics=diagnostics,
        ll=ll,
        group_loglik=group_loglik,
        responsibilities=responsibilities,
        log_responsibilities=log_responsibilities,
        expected_log_membership=expected_log_membership,
        posterior_membership=posterior_membership,
        experience_membership=experience_membership,
        credibility=credibility,
        gate_prob=validated.gate_prob,
        group_ids=validated.group_ids,
        group_index=validated.group_index,
        row_group=validated.row_group,
        X_group=validated.X_group,
    )
end

"""
    dmmm_estep_dp(expert_loglik, X, group, α, ϕ; max_dp_states=2_000_000)

Compute the exact DMMM likelihood, period responsibilities `τ`, and
expected log memberships `ξ` by forward-backward recursion over cumulative
class-count states. This is exact but its state count is combinatorial in the
number of components.
"""
function dmmm_estep_dp(
    expert_loglik,
    X,
    group,
    α,
    ϕ;
    max_dp_states=2_000_000,
)
    max_dp_states isa Integer && max_dp_states >= 1 ||
        throw(ArgumentError("max_dp_states must be a positive integer."))
    validated = _dmmm_validate_estep_inputs(expert_loglik, X, group, α, ϕ)
    responsibilities = zeros(
        Float64, validated.n_observations, validated.n_components
    )
    log_responsibilities=fill(-Inf,size(responsibilities))
    expected_log_membership = zeros(
        Float64, length(validated.group_ids), validated.n_components
    )
    group_loglik = zeros(Float64, length(validated.group_ids))
    state_counts = zeros(Int, length(validated.group_ids))

    for i in eachindex(validated.group_index)
        rows = validated.group_index[i]
        n_periods = length(rows)
        dirichlet_parameter = ϕ .* vec(validated.gate_prob[i, :])
        zero_count = ntuple(unused -> 0, validated.n_components)
        forward = Vector{Dict{Tuple,Float64}}(undef, n_periods + 1)
        forward[1] = Dict{Tuple,Float64}(zero_count => 0.0)
        cumulative_states = 1

        for t in 1:n_periods
            previous = forward[t]
            current = Dict{Tuple,Float64}()
            denominator = log(ϕ + t - 1)
            for (count, log_forward) in previous
                for k in 1:validated.n_components
                    log_emission = expert_loglik[rows[t], k]
                    log_emission == -Inf && continue
                    next_count = _dmmm_increment_count(count, k)
                    value = log_forward +
                            log(dirichlet_parameter[k] + count[k]) - denominator +
                            log_emission
                    current[next_count] = _dmmm_logaddexp(
                        get(current, next_count, -Inf), value
                    )
                end
            end
            isempty(current) &&
                throw(
                    ArgumentError(
                        "Policyholder $(validated.group_ids[i]) has zero likelihood under every class path."
                    ),
                )
            cumulative_states += length(current)
            cumulative_states <= max_dp_states ||
                throw(
                    ArgumentError(
                        "Policyholder $(validated.group_ids[i]) requires more than " *
                        "max_dp_states=$max_dp_states count states. Use method=:mcem " *
                        "or increase max_dp_states when memory permits."
                    ),
                )
            forward[t + 1] = current
        end

        terminal_values = collect(values(forward[end]))
        group_loglik[i] = logsumexp(terminal_values)
        state_counts[i] = cumulative_states

        backward = Vector{Dict{Tuple,Float64}}(undef, n_periods + 1)
        backward[end] = Dict{Tuple,Float64}(
            count => 0.0 for count in keys(forward[end])
        )
        for t in (n_periods - 1):-1:0
            current = Dict{Tuple,Float64}()
            denominator = log(ϕ + t)
            for count in keys(forward[t + 1])
                value = -Inf
                for k in 1:validated.n_components
                    log_emission = expert_loglik[rows[t + 1], k]
                    log_emission == -Inf && continue
                    next_count = _dmmm_increment_count(count, k)
                    haskey(backward[t + 2], next_count) || continue
                    term = log(dirichlet_parameter[k] + count[k]) - denominator +
                           log_emission + backward[t + 2][next_count]
                    value = _dmmm_logaddexp(value, term)
                end
                current[count] = value
            end
            backward[t + 1] = current
        end

        for t in 1:n_periods
            denominator = log(ϕ + t - 1)
            for (count, log_forward) in forward[t]
                for k in 1:validated.n_components
                    log_emission = expert_loglik[rows[t], k]
                    log_emission == -Inf && continue
                    next_count = _dmmm_increment_count(count, k)
                    haskey(backward[t + 1], next_count) || continue
                    log_joint = log_forward +
                                log(dirichlet_parameter[k] + count[k]) - denominator +
                                log_emission + backward[t + 1][next_count]
                    log_responsibilities[rows[t],k]=_dmmm_logaddexp(
                        log_responsibilities[rows[t],k],log_joint-group_loglik[i])
                end
            end
            responsibilities[rows[t],:] .= exp.(view(log_responsibilities,rows[t],:))
        end

        logabsxi=fill(-Inf,validated.n_components)
        for (count, log_forward) in forward[end]
            for k in 1:validated.n_components
                value=digamma(dirichlet_parameter[k]+count[k])-digamma(ϕ+n_periods)
                value<0 && (logabsxi[k]=_dmmm_logaddexp(logabsxi[k],
                    log_forward-group_loglik[i]+log(-value)))
            end
        end
        expected_log_membership[i,:] .= -exp.(logabsxi)
    end

    return _dmmm_finish_estep(
        :dp,
        (state_counts=state_counts, total_states=sum(state_counts)),
        sum(group_loglik),
        group_loglik,
        responsibilities,
        expected_log_membership,
        validated,
        ϕ;log_responsibilities=log_responsibilities,
    )
end

function _dmmm_sample_logweights(rng, logweights)
    largest = maximum(logweights)
    isfinite(largest) ||
        throw(ArgumentError("Every component has zero conditional probability."))
    weights = exp.(logweights .- largest)
    threshold = rand(rng) * sum(weights)
    cumulative = 0.0
    for k in eachindex(weights)
        cumulative += weights[k]
        threshold <= cumulative && return k
    end
    return lastindex(weights)
end

function _dmmm_gibbs_sweep!(rng, classes, counts, loglik, dirichlet_parameter)
    n_periods, n_components = size(loglik)
    logweights = Vector{Float64}(undef, n_components)
    for t in 1:n_periods
        old_component = classes[t]
        counts[old_component] -= 1
        for k in 1:n_components
            logweights[k] =
                log(dirichlet_parameter[k] + counts[k]) + loglik[t, k]
        end
        new_component = _dmmm_sample_logweights(rng, logweights)
        classes[t] = new_component
        counts[new_component] += 1
    end
    return nothing
end

"""
    dmmm_estep_mcem(expert_loglik, X, group, α, ϕ; ...)

Approximate `τ` and `ξ` using the collapsed Gibbs sampler from the DMMM
MCEM algorithm. `mcem_samples` is the number of retained class sequences per
chain. The observed likelihood is not estimated and is returned as `NaN`.
"""
function dmmm_estep_mcem(
    expert_loglik,
    X,
    group,
    α,
    ϕ;
    rng=Random.GLOBAL_RNG,
    mcem_burnin=200,
    mcem_samples=500,
    mcem_thin=1,
    mcem_chains=1,
)
    rng isa AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG."))
    for (name, value, minimum_value) in (
        (:mcem_burnin, mcem_burnin, 0),
        (:mcem_samples, mcem_samples, 1),
        (:mcem_thin, mcem_thin, 1),
        (:mcem_chains, mcem_chains, 1),
    )
        value isa Integer && value >= minimum_value ||
            throw(ArgumentError("$name must be an integer at least $minimum_value."))
    end
    validated = _dmmm_validate_estep_inputs(expert_loglik, X, group, α, ϕ)
    responsibilities = zeros(
        Float64, validated.n_observations, validated.n_components
    )
    expected_log_membership = zeros(
        Float64, length(validated.group_ids), validated.n_components
    )
    total_draws = mcem_samples * mcem_chains
    max_naive_mcse = 0.0

    for i in eachindex(validated.group_index)
        rows = validated.group_index[i]
        n_periods = length(rows)
        loglik = Matrix{Float64}(expert_loglik[rows, :])
        dirichlet_parameter = ϕ .* vec(validated.gate_prob[i, :])
        responsibility_sum = zeros(Float64, n_periods, validated.n_components)
        xi_sum = zeros(Float64, validated.n_components)

        for unused_chain in 1:mcem_chains
            classes = Vector{Int}(undef, n_periods)
            counts = zeros(Int, validated.n_components)
            initial_logweights = Vector{Float64}(undef, validated.n_components)
            for t in 1:n_periods
                for k in 1:validated.n_components
                    initial_logweights[k] =
                        log(validated.gate_prob[i, k]) + loglik[t, k]
                end
                classes[t] = _dmmm_sample_logweights(rng, initial_logweights)
                counts[classes[t]] += 1
            end

            for unused_sweep in 1:mcem_burnin
                _dmmm_gibbs_sweep!(
                    rng, classes, counts, loglik, dirichlet_parameter
                )
            end
            for unused_draw in 1:mcem_samples
                for unused_thin in 1:mcem_thin
                    _dmmm_gibbs_sweep!(
                        rng, classes, counts, loglik, dirichlet_parameter
                    )
                end
                for t in 1:n_periods
                    responsibility_sum[t, classes[t]] += 1.0
                end
                for k in 1:validated.n_components
                    xi_sum[k] +=
                        digamma(dirichlet_parameter[k] + counts[k]) -
                        digamma(ϕ + n_periods)
                end
            end
        end

        responsibilities[rows, :] = responsibility_sum ./ total_draws
        expected_log_membership[i, :] = xi_sum ./ total_draws
        for probability in responsibilities[rows, :]
            max_naive_mcse = max(
                max_naive_mcse,
                sqrt(probability * (1 - probability) / total_draws),
            )
        end
    end

    group_loglik = fill(NaN, length(validated.group_ids))
    return _dmmm_finish_estep(
        :mcem,
        (
            burnin=mcem_burnin,
            samples_per_chain=mcem_samples,
            thin=mcem_thin,
            chains=mcem_chains,
            total_draws=total_draws,
            max_naive_responsibility_mcse=max_naive_mcse,
        ),
        NaN,
        group_loglik,
        responsibilities,
        expected_log_membership,
        validated,
        ϕ,
    )
end

"""
    dmmm_estep(expert_loglik, X, group, α, ϕ; method=:enumeration, ...)

`expert_loglik` may contain ordinary class log densities or precomputed weighted
class scores from `dmmm_expert_logscores`. All backends use those emissions
unchanged; response weights never rescale categorical counts or Dirichlet terms.

Run one of the DMMM E-step backends:

- `:closed_form`: exact coincidence-pattern formulas for `T <= 3`;
- `:enumeration`: exact enumeration of all `K^T` class sequences;
- `:dp`: exact forward-backward recursion over count states;
- `:mcem`: collapsed-Gibbs Monte Carlo approximation;
- `:auto`: closed form for `T <= 3`, enumeration when small, otherwise DP when its state budget is
  feasible, and MCEM otherwise.

The default remains `:enumeration` for backward compatibility. Use
`fit_DMMM(...; estep=:auto)` for automatic fitting.
"""
function dmmm_estep(
    expert_loglik,
    X,
    group,
    α,
    ϕ;
    method=:enumeration,
    max_assignments=1_000_000,
    max_dp_states=2_000_000,
    rng=Random.GLOBAL_RNG,
    mcem_burnin=200,
    mcem_samples=500,
    mcem_thin=1,
    mcem_chains=1,
)
    max_assignments isa Integer && max_assignments >= 1 ||
        throw(ArgumentError("max_assignments must be a positive integer."))
    max_dp_states isa Integer && max_dp_states >= 1 ||
        throw(ArgumentError("max_dp_states must be a positive integer."))
    validated = _dmmm_validate_estep_inputs(expert_loglik, X, group, α, ϕ)
    resolved_method = _dmmm_resolve_estep_method(
        method,
        validated.n_components,
        validated.group_index;
        max_assignments=max_assignments,
        max_dp_states=max_dp_states,
    )
    if resolved_method == :closed_form
        return dmmm_estep_closed_form(expert_loglik, X, group, α, ϕ)
    elseif resolved_method == :enumeration
        return _dmmm_estep_enumeration(
            expert_loglik,
            X,
            group,
            α,
            ϕ;
            max_assignments=max_assignments,
        )
    elseif resolved_method == :dp
        return dmmm_estep_dp(
            expert_loglik, X, group, α, ϕ; max_dp_states=max_dp_states
        )
    end
    return dmmm_estep_mcem(
        expert_loglik,
        X,
        group,
        α,
        ϕ;
        rng=rng,
        mcem_burnin=mcem_burnin,
        mcem_samples=mcem_samples,
        mcem_thin=mcem_thin,
        mcem_chains=mcem_chains,
    )
end

function _dmmm_mcem_q(
    expert_loglik,
    α,
    ϕ,
    X_group,
    responsibilities,
    expected_log_membership,
    model,
    penalty,
    pen_α,
    pen_params,
)
    q_value = 0.0
    for row in axes(expert_loglik, 1), k in axes(expert_loglik, 2)
        weight = responsibilities[row, k]
        weight == 0 && continue
        q_value += weight * expert_loglik[row, k]
    end
    gate_prob = _dmmm_gate_probabilities(α, X_group)
    for i in axes(X_group, 1)
        π_i = vec(gate_prob[i, :])
        ξ_i = vec(expected_log_membership[i, :])
        dirichlet_parameter = ϕ .* π_i
        q_value +=
            loggamma(ϕ) - sum(loggamma.(dirichlet_parameter)) +
            sum((dirichlet_parameter .- 1.0) .* ξ_i)
    end
    return q_value + _dmmm_penalty(
        α, model, penalty, pen_α, pen_params
    )
end

function _dmmm_single_expert_loglik(Y, expert, exposure)
    result = Vector{Float64}(undef, length(Y))
    @threads for row in eachindex(Y)
        exposed = exposurize_expert(expert; exposure=exposure[row])
        result[row] = expert_ll_exact(exposed, Y[row])
    end
    return result
end

function _dmmm_fit_mcem(
    Y,
    X,
    group,
    α_init,
    ϕ_init,
    model;
    exposure,
    response_weights,
    penalty,
    pen_α,
    pen_params,
    ϵ,
    ecm_iter_max,
    dirichlet_iter_max,
    min_ϕ,
    max_ϕ,
    update_alpha,
    update_phi,
    update_experts,
    print_steps,
    rng,
    mcem_burnin,
    mcem_samples,
    mcem_thin,
    mcem_chains,
    mcem_min_iterations,
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
)
    mcem_min_iterations isa Integer && mcem_min_iterations >= 1 ||
        throw(ArgumentError("mcem_min_iterations must be a positive integer."))
    mcem_expert_update in (:coordinate, :batch) || throw(ArgumentError(
        "mcem_expert_update must be :coordinate or :batch."))
    mcem_phi_update in (:q, :ecme) || throw(ArgumentError(
        "mcem_phi_update must be :q or :ecme."))
    mcem_observed_trace isa Bool || throw(ArgumentError(
        "mcem_observed_trace must be a Bool."))
    mcem_ecme_stability_window isa Integer && mcem_ecme_stability_window >= 1 ||
        throw(ArgumentError("mcem_ecme_stability_window must be positive."))
    all(x -> x isa Real && isfinite(x) && x >= 0,
        (mcem_ecme_alpha_tol, mcem_ecme_log_phi_tol,
         mcem_ecme_log_expert_tol, mcem_ecme_expert_min_support)) || throw(ArgumentError(
        "MCEM ECME stability tolerances must be finite and nonnegative."))
    minimum_iterations = min(mcem_min_iterations, ecm_iter_max)
    α_em = _dmmm_canonical_alpha(α_init)
    ϕ_em = Float64(ϕ_init)
    model_em = copy(model)
    expert_loglik = _dmmm_expert_logscores(Y, model_em, exposure, response_weights)

    mcem_phi_init in (:provided, :profile) || throw(ArgumentError("mcem_phi_init must be :provided or :profile."))
    controls = isnothing(mcem_phi_gate) ? nothing : _dmmm_phi_controls(mcem_phi_gate)
    profile_enabled = mcem_phi_init == :profile || !isnothing(controls) || mcem_phi_update == :ecme
    validated = profile_enabled ? _dmmm_validate_estep_inputs(expert_loglik, X, group, α_em, ϕ_em) : nothing
    if profile_enabled
        update_phi || throw(ArgumentError("Profile initialization/gating requires update_phi=true."))
        all(rows -> length(rows) <= 3, validated.group_index) || throw(ArgumentError(
            "MCEM phi profiles currently support at most three observations per group; no exact E-step fallback is used."))
        any(rows -> length(rows) > 1, validated.group_index) || throw(ArgumentError(
            "Phi is unidentified when every group has one observation."))
    end
    phi_events, history = NamedTuple[], NamedTuple[]
    initial_profile = nothing
    supplied_phi = ϕ_em
    if mcem_phi_init == :profile
        terms = _dmmm_phi_terms(expert_loglik, validated.X_group, validated.group_index, α_em)
        initial_profile = _dmmm_phi_profile(terms, ϕ_em; min_phi=min_ϕ, max_phi=max_ϕ,
            grid_size=isnothing(controls) ? 81 : controls.grid_size)
        ϕ_em = initial_profile.phi_profile
    end
    initialized_phi = ϕ_em
    if profile_enabled && !isnothing(mcem_callback)
        mcem_callback((iteration=0, alpha=copy(α_em), phi=ϕ_em,
            model=deepcopy(model_em), q_before=NaN, q_after=NaN, conditional_increment=NaN))
    end
    rescue_count, gross_count, stable_count, mc_factor, last_reset = 0, 0, 0, 1, 0
    stop_reason = :iteration_limit
    active_samples, active_burnin = mcem_samples, mcem_burnin

    run_estep = (loglik, α, ϕ; estep_rng=rng) -> dmmm_estep_mcem(
        loglik,
        X,
        group,
        α,
        ϕ;
        rng=estep_rng,
        mcem_burnin=active_burnin,
        mcem_samples=active_samples,
        mcem_thin=mcem_thin,
        mcem_chains=mcem_chains,
    )
    e_step = run_estep(expert_loglik, α_em, ϕ_em)
    repeated_groups = any(length(rows) > 1 for rows in e_step.group_index)
    update_phi && !repeated_groups &&
        @warn(
            "All policyholders have one observation. The marginal model does not identify ϕ; keeping ϕ at its initial value."
        )

    objective_trace = Float64[]
    observed_trace = Float64[]
    batch_acceptances = 0
    coordinate_fallbacks = 0
    ecme_parameter_steps = NamedTuple[]
    previous_alpha = copy(α_em)
    previous_phi = ϕ_em
    previous_experts = _dmmm_parameter_snapshot(model_em).values
    expert_components = _dmmm_parameter_snapshot(model_em).components
    converge = false
    iteration = 0
    for current_iteration in 1:ecm_iter_max
        iteration = current_iteration
        responsibilities = e_step.responsibilities
        expected_log_membership = e_step.expected_log_membership
        q_before = _dmmm_mcem_q(
            expert_loglik,
            α_em,
            ϕ_em,
            e_step.X_group,
            responsibilities,
            expected_log_membership,
            model_em,
            penalty,
            pen_α,
            pen_params,
        )
        q_current = q_before
        batch_fallback = false

        if update_experts && mcem_expert_update == :batch
            previous_model = deepcopy(model_em)
            candidate_model = deepcopy(model_em)
            for j in axes(candidate_model, 1), k in axes(candidate_model, 2)
                response_weights[j] == 0 && continue
                candidate_model[j, k] = try
                    EM_M_expert_exact(
                        model_em[j, k], Y[:, j], exposure,
                        response_weights[j] .* vec(responsibilities[:, k]);
                        penalty=penalty, pen_pararms_jk=pen_params[j][k],
                    )
                catch error
                    @warn("Expert update failed; retaining the previous expert.",
                        response_dimension=j, component=k,
                        exception=(error, catch_backtrace()))
                    model_em[j, k]
                end
            end
            candidate_loglik = _dmmm_expert_logscores(
                Y, candidate_model, exposure, response_weights)
            candidate_q = _dmmm_mcem_q(candidate_loglik, α_em, ϕ_em,
                e_step.X_group, responsibilities, expected_log_membership,
                candidate_model, penalty, pen_α, pen_params)
            if isfinite(candidate_q) && candidate_q >= q_current - 1e-8
                model_em = candidate_model
                expert_loglik = candidate_loglik
                q_current = candidate_q
                batch_acceptances += 1
            else
                coordinate_fallbacks += 1
                batch_fallback = true
                model_em = previous_model
            end
        end

        if update_experts && (mcem_expert_update == :coordinate ||
                              batch_fallback)
            for j in axes(model_em, 1), k in axes(model_em, 2)
                response_weights[j] == 0 && continue
                previous_expert = model_em[j, k]
                previous_contribution = _dmmm_single_expert_loglik(
                    view(Y, :, j), previous_expert, exposure
                )
                candidate = try
                    EM_M_expert_exact(
                        previous_expert,
                        Y[:, j],
                        exposure,
                        response_weights[j] .* vec(responsibilities[:, k]);
                        penalty=penalty,
                        pen_pararms_jk=pen_params[j][k],
                    )
                catch error
                    @warn(
                        "Expert update failed; retaining the previous expert.",
                        response_dimension=j,
                        component=k,
                        exception=(error, catch_backtrace()),
                    )
                    previous_expert
                end
                candidate_contribution = _dmmm_single_expert_loglik(
                    view(Y, :, j), candidate, exposure
                )
                contribution_change = response_weights[j] .*
                                      (candidate_contribution .- previous_contribution)
                penalty_before = _dmmm_penalty(
                    α_em, model_em, penalty, pen_α, pen_params
                )
                model_em[j, k] = candidate
                penalty_after = _dmmm_penalty(
                    α_em, model_em, penalty, pen_α, pen_params
                )
                likelihood_change = 0.0
                for row in axes(Y, 1)
                    weight = responsibilities[row, k]
                    weight == 0 && continue
                    likelihood_change += weight * contribution_change[row]
                end
                candidate_q = q_current + likelihood_change +
                              (penalty_after - penalty_before)
                if isfinite(candidate_q) && candidate_q >= q_current - 1e-8
                    expert_loglik[:, k] .+= contribution_change
                    q_current = candidate_q
                else
                    model_em[j, k] = previous_expert
                end
            end
        end

        update_phi_now = update_phi && repeated_groups && mcem_phi_update == :q
        α_candidate, ϕ_candidate = try
            _dmmm_update_dirichlet(
                α_em,
                ϕ_em,
                e_step.X_group,
                expected_log_membership;
                update_alpha=update_alpha,
                update_phi=update_phi_now,
                penalty=penalty,
                pen_α=pen_α,
                dirichlet_iter_max=dirichlet_iter_max,
                min_ϕ=min_ϕ,
                max_ϕ=max_ϕ,
            )
        catch error
            @warn(
                "Dirichlet-regression update failed; retaining α and ϕ.",
                exception=(error, catch_backtrace()),
            )
            α_em, ϕ_em
        end
        candidate_q = _dmmm_mcem_q(
            expert_loglik,
            α_candidate,
            ϕ_candidate,
            e_step.X_group,
            responsibilities,
            expected_log_membership,
            model_em,
            penalty,
            pen_α,
            pen_params,
        )
        if isfinite(candidate_q) && candidate_q >= q_current - 1e-8
            α_em = α_candidate
            ϕ_em = ϕ_candidate
            q_current = candidate_q
        end

        if mcem_phi_update == :ecme && update_phi && repeated_groups
            terms = _dmmm_phi_terms(
                expert_loglik, e_step.X_group, e_step.group_index, α_em)
            profile = _dmmm_phi_profile(
                terms, ϕ_em; min_phi=min_ϕ, max_phi=max_ϕ,
                grid_size=isnothing(controls) ? 81 : controls.grid_size)
            ϕ_em = profile.phi_profile
            q_current = _dmmm_mcem_q(expert_loglik, α_em, ϕ_em,
                e_step.X_group, responsibilities, expected_log_membership,
                model_em, penalty, pen_α, pen_params)
        end

        improvement = q_current - q_before
        push!(objective_trace, q_current)
        if mcem_phi_update == :ecme || mcem_observed_trace
            terms = _dmmm_phi_terms(
                expert_loglik, e_step.X_group, e_step.group_index, α_em)
            push!(observed_trace, _dmmm_phi_value(terms, ϕ_em).value)
        end
        if mcem_phi_update == :ecme
            current_experts = _dmmm_parameter_snapshot(model_em).values
            support = vec(sum(responsibilities; dims=1))
            selected_experts = [support[k] >= mcem_ecme_expert_min_support
                                for k in expert_components]
            push!(ecme_parameter_steps, (
                alpha=maximum(abs.(α_em .- previous_alpha)),
                log_phi=abs(log(ϕ_em / previous_phi)),
                log_expert=any(selected_experts) ? maximum(abs.(log.(
                    current_experts[selected_experts] ./ previous_experts[selected_experts]))) : Inf,
            ))
            previous_alpha .= α_em
            previous_phi = ϕ_em
            previous_experts = current_experts
        end
        if !isnothing(mcem_callback)
            mcem_callback((
                iteration=current_iteration, alpha=copy(α_em), phi=ϕ_em,
                model=deepcopy(model_em), q_before=q_before, q_after=q_current,
                conditional_increment=improvement,
                observed_criterion=isempty(observed_trace) ? NaN : observed_trace[end],
                responsibilities=copy(responsibilities),
                posterior_membership=copy(e_step.posterior_membership),
                gate_prob=copy(e_step.gate_prob),
                batch_acceptances=batch_acceptances,
                coordinate_fallbacks=coordinate_fallbacks,
            ))
        end
        print_steps > 0 && current_iteration % print_steps == 0 &&
            @info(
                "DMMM MCEM iteration $(current_iteration): Q $(q_current), " *
                "conditional increment $(improvement), ϕ $(ϕ_em)"
            )

        ordinary_stop = current_iteration >= minimum_iterations && -1e-8 <= improvement <= ϵ
        if !isnothing(controls)
            terms = _dmmm_phi_terms(expert_loglik, e_step.X_group, e_step.group_index, α_em)
            observed = _dmmm_phi_value(terms, ϕ_em).value +
                _dmmm_penalty(α_em, model_em, penalty, pen_α, pen_params)
            parameter_snapshot = _dmmm_parameter_snapshot(model_em)
            push!(history, (phi=ϕ_em, alpha=copy(α_em),
                parameters=parameter_snapshot.values,
                parameter_components=parameter_snapshot.components,
                support=vec(sum(responsibilities; dims=1)), criterion=observed))
            length(history) > controls.window+1 && popfirst!(history)
            scheduled = current_iteration % controls.check_interval == 0 ||
                current_iteration == ecm_iter_max
            if scheduled
                profile = _dmmm_phi_profile(terms, ϕ_em; min_phi=min_ϕ,
                    max_phi=max_ϕ, grid_size=controls.grid_size)
                stability = _dmmm_phi_stability(history, controls, e_step.X_group, size(Y, 1))
                gain_per_group = profile.criterion_gain / length(e_step.group_ids)
                score_per_group = profile.log_phi_score / length(e_step.group_ids)
                credibility_current = controls.credibility_T /
                    (controls.credibility_T + profile.phi_current)
                credibility_profile = controls.credibility_T /
                    (controls.credibility_T + profile.phi_profile)
                credibility_difference = abs(credibility_current - credibility_profile)
                profile_pass = !profile.flat && !profile.at_bound &&
                    gain_per_group <= controls.gain_tol_per_group &&
                    credibility_difference <= controls.credibility_tol
                mc_sd, mc_seed = NaN, UInt(0)
                mc_values = Float64[]
                action = :monitor
                ready = ordinary_stop && current_iteration-last_reset >= controls.window
                hard_pass = ready && profile_pass && stability.phi_stable
                !hard_pass && (stable_count = 0)
                gross = !profile.flat && !profile.at_bound &&
                    profile.log_phi_distance >= controls.rescue_log_distance &&
                    gain_per_group >= controls.rescue_gain_per_group
                gross_count = gross ? gross_count + 1 : 0
                if controls.rescue == :damped && gross_count >= controls.rescue_windows &&
                   current_iteration-last_reset >= controls.window
                    if rescue_count >= controls.max_rescues
                        action = :rescue_limit
                    elseif ecm_iter_max-current_iteration < controls.window+1
                        action = :rescue_pending
                    else
                        # Optional rescue on the log scale. This is not a Q M-step
                        # and never runs in the default :none configuration.
                        ϕ_em = exp((1-controls.rescue_rho)*log(ϕ_em) +
                            controls.rescue_rho*log(profile.phi_profile))
                        rescue_count += 1
                        gross_count, last_reset, stable_count = 0, current_iteration, 0
                        empty!(history)
                        action = :damped_rescue
                    end
                elseif hard_pass
                    if controls.mc_diagnostic
                        # This estimates conditional phi-update variability for
                        # reporting. It is not a stopping gate by default.
                        mc_seed = rand(rng, UInt)
                        diagnostic_rng = MersenneTwister(mc_seed)
                        for unused in 1:controls.mc_replicates
                            replicate = run_estep(expert_loglik, α_em, ϕ_em; estep_rng=diagnostic_rng)
                            unused_alpha, p = _dmmm_update_dirichlet(α_em, ϕ_em,
                                replicate.X_group, replicate.expected_log_membership;
                                update_alpha=update_alpha, update_phi=true, penalty=penalty,
                                pen_α=pen_α, dirichlet_iter_max=dirichlet_iter_max,
                                min_ϕ=min_ϕ, max_ϕ=max_ϕ)
                            push!(mc_values, p)
                        end
                        logs = log.(mc_values)
                        mc_sd = sqrt(sum(abs2, logs .- sum(logs)/length(logs))/(length(logs)-1))
                    end
                    stable_count += 1
                    if stable_count >= controls.stable_checks
                        converge, action, stop_reason = true, :converged, :profile_converged
                    else
                        action = :confirm_stability
                    end
                else
                    action = gross ? :gross_discrepancy : :monitor
                end
                push!(phi_events, merge((iteration=current_iteration, ordinary_stop=ordinary_stop),
                    profile, stability, (criterion_gain_per_group=gain_per_group,
                    log_phi_score_per_group=score_per_group, profile_pass=profile_pass,
                    credibility_current=credibility_current,
                    credibility_profile=credibility_profile,
                    credibility_difference=credibility_difference,
                    hard_convergence_pass=hard_pass,
                    gross_discrepancy=gross, gross_windows=gross_count,
                    mc_log_phi_sd=mc_sd,
                    mc_seed=mc_seed, mc_phi_updates=mc_values, mc_factor=mc_factor,
                    rescue_count=rescue_count, action=action)))
                print_steps > 0 && @info("MCEM phi gate", iteration=current_iteration,
                    phi=profile.phi_current, profile_phi=profile.phi_profile,
                    gain=profile.criterion_gain, score=profile.log_phi_score, action=action)
            end
        elseif ordinary_stop
            if mcem_phi_update == :q
                converge, stop_reason = true, :ordinary_converged
            end
        end
        if mcem_phi_update == :ecme && current_iteration >= minimum_iterations &&
           length(observed_trace) >= mcem_ecme_stability_window + 1
            recent_objective = observed_trace[end-mcem_ecme_stability_window:end]
            recent_steps = ecme_parameter_steps[end-mcem_ecme_stability_window+1:end]
            if maximum(abs.(diff(recent_objective))) <= ϵ &&
               maximum(x.alpha for x in recent_steps) <= mcem_ecme_alpha_tol &&
               maximum(x.log_phi for x in recent_steps) <= mcem_ecme_log_phi_tol &&
               maximum(x.log_expert for x in recent_steps) <= mcem_ecme_log_expert_tol
                converge, stop_reason = true, :ecme_observed_converged
            end
        end
        # Refresh after every accepted move, including any profile restart.
        e_step = run_estep(expert_loglik, α_em, ϕ_em)
        converge && break
    end

    if update_phi && repeated_groups &&
       (isapprox(ϕ_em, min_ϕ; rtol=1e-5) ||
        isapprox(ϕ_em, max_ϕ; rtol=1e-5))
        @warn(
            "The fitted ϕ is at an optimization bound; within-policyholder dependence may be weakly identified.",
            ϕ=ϕ_em,
        )
    end
    return (
        α_fit=α_em,
        ϕ_fit=ϕ_em,
        model_fit=model_em,
        converge=converge,
        iter=iteration,
        ll_np=NaN,
        ll=NaN,
        AIC=NaN,
        BIC=NaN,
        group_ids=e_step.group_ids,
        responsibilities=e_step.responsibilities,
        posterior_membership=e_step.posterior_membership,
        loglik_trace=mcem_phi_update == :ecme ? observed_trace : objective_trace,
        estep_method=:mcem,
        estep_diagnostics=merge(e_step.diagnostics, (phi_initialization=mcem_phi_init,
            supplied_phi=supplied_phi, initialized_phi=initialized_phi,
            initial_phi_profile=initial_profile, phi_gate_controls=controls,
            phi_gate_events=phi_events, phi_restart_count=rescue_count,
            phi_rescue_count=rescue_count,
            stop_reason=stop_reason, expert_update=mcem_expert_update,
            phi_update=mcem_phi_update, batch_acceptances=batch_acceptances,
            coordinate_fallbacks=coordinate_fallbacks,
            observed_trace=observed_trace,
            ecme_parameter_steps=ecme_parameter_steps)),
        objective_name=mcem_phi_update == :ecme ?
            (_dmmm_is_weighted(response_weights) ? :weighted_loglik : :loglik) :
            (_dmmm_is_weighted(response_weights) ? :weighted_Q : :Q),
    )
end
