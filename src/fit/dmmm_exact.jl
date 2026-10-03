const _DMMM_GATE_FLOOR = 1e-12
const _DMMM_POLICY_CHUNKS = 64

"""Fixed policy ranges keep threaded reductions reproducible across thread counts."""
function _dmmm_policy_ranges(n_items; max_chunks=_DMMM_POLICY_CHUNKS)
    n_items >= 0 || throw(ArgumentError("n_items must be nonnegative."))
    max_chunks >= 1 || throw(ArgumentError("max_chunks must be positive."))
    n_items == 0 && return UnitRange{Int}[]
    chunk_width = cld(n_items, min(n_items, max_chunks))
    return [first:min(first + chunk_width - 1, n_items) for first in 1:chunk_width:n_items]
end

function _dmmm_log_rising_factorial(value, count)
    count == 0 && return 0.0
    result = 0.0
    for offset in 0:(count - 1)
        result += log(value + offset)
    end
    return result
end

function _dmmm_fit_exact_block(
    Y, X, group, α_init, ϕ_init, model;
    exposure, response_weights, penalty, pen_α, pen_params, ϵ,
    ecm_iter_max, dirichlet_iter_max, min_ϕ, max_ϕ,
    estep_method, max_assignments, max_dp_states,
    update_alpha, update_phi, update_experts, print_steps,
    zigamma_update_tracker, iteration_offset,
    exact_phi_update, exact_callback,
)
    α_em = _dmmm_canonical_alpha(α_init)
    ϕ_em = Float64(ϕ_init)
    model_em = copy(model)
    expert_loglik = _dmmm_expert_logscores(Y, model_em, exposure, response_weights)
    run_estep(scores, alpha, phi) = dmmm_estep(
        scores, X, group, alpha, phi; method=estep_method,
        max_assignments=max_assignments, max_dp_states=max_dp_states)
    e_step = run_estep(expert_loglik, α_em, ϕ_em)
    e_step_calls = 1
    repeated_groups = any(length(rows) > 1 for rows in e_step.group_index)
    exact_phi_update == :ecme && update_phi &&
        maximum(length, e_step.group_index) > 3 && throw(ArgumentError(
            "Exact ECME phi profiling supports at most three observations per group."))

    ll_np = e_step.ll
    ll = ll_np + _dmmm_penalty(α_em, model_em, penalty, pen_α, pen_params)
    loglik_trace = [ll]
    phi_trace = [ϕ_em]
    expert_block_trace = NamedTuple[]
    gate_phi_block_trace = NamedTuple[]
    diagnostic_failures = 0
    gate_backtracks = 0
    converge = false
    stop_reason = :iteration_limit
    iteration = 0

    update_phi && !repeated_groups && @warn(
        "All policyholders have one observation. The marginal criterion does not identify ϕ; keeping ϕ at its initial value.")
    objective_label = _dmmm_is_weighted(response_weights) ?
        "integrated weighted criterion" : "observed loglikelihood"
    print_steps > 0 && @info(
        "Initial block exact DMMM $objective_label: $ll")

    for current_iteration in 1:ecm_iter_max
        iteration = current_iteration
        ll_old = ll
        decrease_tolerance = max(1e-8, 1e-10 * max(abs(ll_old), 1.0))
        failure_this_iteration = false

        # All experts use the same cached E-step quantities.
        expert_before = ll
        if update_experts
            candidate_model = copy(model_em)
            for j in axes(candidate_model, 1), k in axes(candidate_model, 2)
                response_weights[j] == 0 && continue
                candidate_model[j, k] = try
                    _em_m_expert_exact_tracked(
                        model_em[j, k], Y[:, j], exposure,
                        response_weights[j] .* vec(e_step.responsibilities[:, k]);
                        penalty=penalty, pen_pararms_jk=pen_params[j][k],
                        tracker=zigamma_update_tracker, diagnostic_key=(j, k),
                        iteration=iteration_offset + current_iteration)
                catch error
                    @warn("Expert update failed; retaining the previous expert.",
                        response_dimension=j, component=k,
                        exception=(error, catch_backtrace()))
                    model_em[j, k]
                end
            end
            candidate_scores = _dmmm_expert_logscores(
                Y, candidate_model, exposure, response_weights)
            candidate_e_step = run_estep(candidate_scores, α_em, ϕ_em)
            e_step_calls += 1
            candidate_ll_np = candidate_e_step.ll
            candidate_ll = candidate_ll_np +
                _dmmm_penalty(α_em, candidate_model, penalty, pen_α, pen_params)
            accepted = isfinite(candidate_ll) &&
                       candidate_ll >= ll - decrease_tolerance
            if accepted
                model_em, expert_loglik, e_step =
                    candidate_model, candidate_scores, candidate_e_step
                ll_np, ll = candidate_ll_np, candidate_ll
            else
                diagnostic_failures += 1
                failure_this_iteration = true
                @warn("Exact expert block decreased the integrated criterion; rejecting the whole block.",
                    iteration=current_iteration, before=ll, after=candidate_ll,
                    difference=candidate_ll - ll)
            end
            push!(expert_block_trace, (before=expert_before, after=candidate_ll,
                difference=candidate_ll - expert_before, accepted=accepted))
        else
            push!(expert_block_trace, (before=ll, after=ll, difference=0.0, accepted=true))
        end

        # Update the complete gate block from the post-expert E-step.
        gate_before = ll
        α_candidate, unused_phi = try
            _dmmm_update_dirichlet(
                α_em, ϕ_em, e_step.X_group, e_step.expected_log_membership;
                update_alpha=update_alpha, update_phi=false, penalty=penalty,
                pen_α=pen_α, dirichlet_iter_max=dirichlet_iter_max,
                min_ϕ=min_ϕ, max_ϕ=max_ϕ)
        catch error
            diagnostic_failures += 1
            failure_this_iteration = true
            @warn("Exact gate block update failed; retaining the previous gates.",
                iteration=current_iteration, exception=(error, catch_backtrace()))
            α_em, ϕ_em
        end

        if exact_phi_update == :ecme
            function profiled_gate(candidate_alpha)
                terms = _dmmm_phi_terms(
                    expert_loglik, e_step.X_group, e_step.group_index, candidate_alpha)
                if update_phi && repeated_groups
                    profile = _dmmm_phi_profile(
                        terms, ϕ_em; min_phi=min_ϕ, max_phi=max_ϕ)
                    return profile.phi_profile,
                           profile.criterion + profile.criterion_gain
                end
                return ϕ_em, _dmmm_phi_value(terms, ϕ_em).value
            end
            ϕ_candidate, profiled_ll_np = profiled_gate(α_candidate)
            profiled_ll = profiled_ll_np +
                _dmmm_penalty(α_candidate, model_em, penalty, pen_α, pen_params)
            backtracks = 0
            raw_alpha = α_candidate
            while (!isfinite(profiled_ll) ||
                   profiled_ll < ll - decrease_tolerance) && backtracks < 12
                backtracks += 1
                fraction = 0.5^backtracks
                α_candidate = α_em .+ fraction .* (raw_alpha .- α_em)
                α_candidate[end, :] .= 0.0
                ϕ_candidate, profiled_ll_np = profiled_gate(α_candidate)
                profiled_ll = profiled_ll_np +
                    _dmmm_penalty(α_candidate, model_em, penalty, pen_α, pen_params)
            end
            gate_backtracks += backtracks
            candidate_e_step = run_estep(expert_loglik, α_candidate, ϕ_candidate)
            e_step_calls += 1
        else
            backtracks = 0
            # Q-based phi needs expectations refreshed at the updated gates.
            gate_e_step = run_estep(expert_loglik, α_candidate, ϕ_em)
            e_step_calls += 1
            unused_alpha, ϕ_candidate = _dmmm_update_dirichlet(
                α_candidate, ϕ_em, gate_e_step.X_group,
                gate_e_step.expected_log_membership;
                update_alpha=false, update_phi=update_phi && repeated_groups,
                penalty=penalty, pen_α=pen_α,
                dirichlet_iter_max=dirichlet_iter_max,
                min_ϕ=min_ϕ, max_ϕ=max_ϕ)
            candidate_e_step = run_estep(expert_loglik, α_candidate, ϕ_candidate)
            e_step_calls += 1
        end
        candidate_ll_np = candidate_e_step.ll
        candidate_ll = candidate_ll_np +
            _dmmm_penalty(α_candidate, model_em, penalty, pen_α, pen_params)
        accepted = isfinite(candidate_ll) &&
                   candidate_ll >= ll - decrease_tolerance
        if accepted
            α_em, ϕ_em, e_step = α_candidate, ϕ_candidate, candidate_e_step
            ll_np, ll = candidate_ll_np, candidate_ll
        else
            diagnostic_failures += 1
            failure_this_iteration = true
            @warn("Exact gate/phi block decreased the integrated criterion; rejecting the whole block.",
                iteration=current_iteration, before=ll, after=candidate_ll,
                difference=candidate_ll - ll)
        end
        push!(gate_phi_block_trace, (before=gate_before, after=candidate_ll,
            difference=candidate_ll - gate_before, accepted=accepted,
            backtracks=backtracks))

        push!(loglik_trace, ll)
        push!(phi_trace, ϕ_em)
        improvement = ll - ll_old
        if !isnothing(exact_callback)
            exact_callback((iteration=current_iteration, alpha=copy(α_em), phi=ϕ_em,
                model=deepcopy(model_em), observed_criterion=ll,
                responsibilities=copy(e_step.responsibilities),
                posterior_membership=copy(e_step.posterior_membership),
                expert_block=expert_block_trace[end],
                gate_phi_block=gate_phi_block_trace[end],
                e_step_calls=e_step_calls))
        end
        if print_steps > 0 && current_iteration % print_steps == 0
            @info(
                "Block exact DMMM iteration $current_iteration: $objective_label $ll, increment $improvement, ϕ $ϕ_em, E-step calls $e_step_calls")
            flush(stderr)
        end
        if improvement <= ϵ
            converge = improvement >= -decrease_tolerance && !failure_this_iteration
            stop_reason = failure_this_iteration ? :diagnostic_failure :
                converge ? :observed_criterion_converged : :criterion_decrease
            break
        end
    end

    n_parameters = _count_α(α_em) + _count_params(model_em) + 1
    n_groups = length(e_step.group_ids)
    weighted = _dmmm_is_weighted(response_weights)
    diagnostics = merge(e_step.diagnostics, (
        update_mode=:block, phi_update=exact_phi_update,
        e_step_calls=e_step_calls, expert_block_trace=expert_block_trace,
        gate_phi_block_trace=gate_phi_block_trace, phi_trace=phi_trace,
        gate_backtracks=gate_backtracks,
        diagnostic_failures=diagnostic_failures, stop_reason=stop_reason,
        zigamma_update_summary=isnothing(zigamma_update_tracker) ? NamedTuple[] :
            zigamma_update_summary(zigamma_update_tracker)))
    return (
        α_fit=α_em, ϕ_fit=ϕ_em, model_fit=model_em, converge=converge,
        iter=iteration, ll_np=ll_np, ll=ll,
        AIC=weighted ? NaN : -2.0 * ll_np + 2.0 * n_parameters,
        BIC=weighted ? NaN : -2.0 * ll_np + log(n_groups) * n_parameters,
        group_ids=e_step.group_ids, responsibilities=e_step.responsibilities,
        posterior_membership=e_step.posterior_membership,
        loglik_trace=loglik_trace, estep_method=e_step.method,
        estep_diagnostics=diagnostics,
        objective_name=weighted ? :weighted_loglik : :loglik)
end

function _dmmm_groups(group)
    group_ids = Any[]
    group_lookup = Dict{Any,Int}()
    group_index = Vector{Vector{Int}}()
    row_group = Vector{Int}(undef, length(group))

    for (row, id) in enumerate(group)
        ismissing(id) && throw(ArgumentError("group identifiers cannot be missing."))
        if !haskey(group_lookup, id)
            push!(group_ids, id)
            push!(group_index, Int[])
            group_lookup[id] = length(group_ids)
        end
        index = group_lookup[id]
        push!(group_index[index], row)
        row_group[row] = index
    end

    return group_ids, group_index, row_group
end

function _dmmm_group_covariates(X, group_index, n_observations)
    n_groups = length(group_index)
    if size(X, 1) == n_groups
        return Matrix{Float64}(X)
    elseif size(X, 1) != n_observations
        throw(
            DimensionMismatch(
                "X must have either one row per observation or one row per policyholder."
            ),
        )
    end

    X_group = Matrix{Float64}(undef, n_groups, size(X, 2))
    for i in 1:n_groups
        rows = group_index[i]
        X_group[i, :] = X[first(rows), :]
        for row in Iterators.drop(rows, 1)
            isapprox(X[row, :], X_group[i, :]; rtol=1e-10, atol=1e-12) ||
                throw(
                    ArgumentError(
                        "Baseline covariates must be constant within each policyholder. " *
                        "Put period-specific exposure in the exposure argument."
                    ),
                )
        end
    end
    return X_group
end

function _dmmm_canonical_alpha(α)
    α_new = Matrix{Float64}(α)
    α_new .-= α_new[end, :]'
    α_new[end, :] .= 0.0
    return α_new
end

function _dmmm_gate_probabilities(α, X)
    prob = exp.(LogitGating(α, X))
    all(isfinite, prob) ||
        throw(
            ArgumentError(
                "The gating probabilities are non-finite. Check or rescale X and α."
            ),
        )
    prob .= max.(prob, _DMMM_GATE_FLOOR)
    prob ./= sum(prob; dims=2)
    return prob
end

function _dmmm_assignment_table(n_components, n_periods, max_assignments)
    n_states = 1
    for unused in 1:n_periods
        n_states > div(max_assignments, n_components) &&
            throw(
                ArgumentError(
                    "A policyholder requires more than max_assignments=$max_assignments " *
                    "latent assignments. Increase max_assignments only when exact enumeration " *
                    "is computationally feasible."
                ),
            )
        n_states *= n_components
    end

    assignments = Matrix{Int}(undef, n_states, n_periods)
    for state in 0:(n_states - 1)
        remainder = state
        for t in 1:n_periods
            assignments[state + 1, t] = rem(remainder, n_components) + 1
            remainder = div(remainder, n_components)
        end
    end
    return assignments
end

"""
    _dmmm_estep_enumeration(expert_loglik, X, group, α, ϕ;
                                max_assignments=1_000_000)

Evaluate the exact grouped observed likelihood and Dirichlet E-step in log
space. `expert_loglik[r, k]` is the log density of observation `r` under
component `k`. The returned posterior membership is

`(ϕ * π_i + sum_t τ_it) / (ϕ + T_i)`.

The computation enumerates latent assignments and is intended for short
panels. It is exact rather than variational or Monte Carlo.
"""
function _dmmm_estep_enumeration(
    expert_loglik,
    X,
    group,
    α,
    ϕ;
    max_assignments=1_000_000,
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
    max_assignments isa Integer && max_assignments >= n_components ||
        throw(ArgumentError("max_assignments must be an integer at least as large as K."))

    group_ids, group_index, row_group = _dmmm_groups(group)
    X_group = _dmmm_group_covariates(X, group_index, n_observations)
    size(α, 2) == size(X_group, 2) ||
        throw(DimensionMismatch("The number of columns of α and X must agree."))

    α_ref = _dmmm_canonical_alpha(α)
    gate_prob = _dmmm_gate_probabilities(α_ref, X_group)
    responsibilities = zeros(Float64, n_observations, n_components)
    log_responsibilities=fill(-Inf,n_observations,n_components)
    expected_log_membership = zeros(Float64, length(group_ids), n_components)
    posterior_membership = zeros(Float64, length(group_ids), n_components)
    experience_membership = zeros(Float64, length(group_ids), n_components)
    credibility = zeros(Float64, length(group_ids))
    group_loglik = zeros(Float64, length(group_ids))
    assignment_cache = Dict{Int,Matrix{Int}}()

    for i in eachindex(group_index)
        rows = group_index[i]
        n_periods = length(rows)
        assignments = get!(
            () -> _dmmm_assignment_table(
                n_components, n_periods, max_assignments
            ),
            assignment_cache,
            n_periods,
        )
        logweights = Vector{Float64}(undef, size(assignments, 1))
        counts = zeros(Int, size(assignments, 1), n_components)
        dirichlet_parameter = ϕ .* vec(gate_prob[i, :])
        beta_constant = -_dmmm_log_rising_factorial(ϕ, n_periods)

        for state in axes(assignments, 1)
            logweight = beta_constant
            for t in 1:n_periods
                component = assignments[state, t]
                counts[state, component] += 1
                logweight += expert_loglik[rows[t], component]
            end
            for k in 1:n_components
                logweight += _dmmm_log_rising_factorial(
                    dirichlet_parameter[k], counts[state, k]
                )
            end
            logweights[state] = logweight
        end

        group_loglik[i] = logsumexp(logweights)
        isfinite(group_loglik[i]) ||
            throw(
                ArgumentError(
                    "Policyholder $(group_ids[i]) has zero or non-finite likelihood under every assignment."
                ),
            )
        logabsxi=fill(-Inf,n_components)
        for state in axes(assignments, 1)
            logweight=logweights[state]-group_loglik[i]
            for t in 1:n_periods
                k=assignments[state,t]
                log_responsibilities[rows[t],k]=_dmmm_logaddexp(log_responsibilities[rows[t],k],logweight)
            end
            for k in 1:n_components
                value=digamma(dirichlet_parameter[k]+counts[state,k])-digamma(ϕ+n_periods)
                value<0 && (logabsxi[k]=_dmmm_logaddexp(logabsxi[k],logweight+log(-value)))
            end
        end
        responsibilities[rows,:] .= exp.(log_responsibilities[rows,:])
        expected_log_membership[i,:] .= -exp.(logabsxi)

        expected_counts = vec(sum(responsibilities[rows, :]; dims=1))
        experience_membership[i, :] = expected_counts ./ n_periods
        credibility[i] = n_periods / (ϕ + n_periods)
        posterior_membership[i, :] =
            (dirichlet_parameter .+ expected_counts) ./ (ϕ + n_periods)
    end

    return (
        method=:enumeration,
        diagnostics=(
            cached_assignments=sum(
                size(table, 1) for table in values(assignment_cache)
            ),
        ),
        ll=sum(group_loglik),
        group_loglik=group_loglik,
        responsibilities=responsibilities,
        log_responsibilities=log_responsibilities,
        expected_log_membership=expected_log_membership,
        posterior_membership=posterior_membership,
        experience_membership=experience_membership,
        credibility=credibility,
        gate_prob=gate_prob,
        group_ids=group_ids,
        group_index=group_index,
        row_group=row_group,
        X_group=X_group,
    )
end

function _dmmm_neg_q!(
    gradient,
    values,
    α_current,
    ϕ_current,
    X_group,
    expected_log_membership;
    update_alpha,
    update_phi,
    penalty,
    pen_α,
    row_weights=nothing,
)
    n_components, n_covariates = size(α_current)
    n_alpha = update_alpha ? (n_components - 1) * n_covariates : 0

    α = copy(α_current)
    if update_alpha
        α[1:(n_components - 1), :] .=
            reshape(view(values, 1:n_alpha), n_components - 1, n_covariates)
    end
    α[end, :] .= 0.0
    ϕ = update_phi ? exp(values[end]) : ϕ_current
    need_gradient = gradient !== nothing
    gate_prob = _dmmm_gate_probabilities(α, X_group)
    raw_gate_prob = need_gradient && update_alpha ? exp.(LogitGating(α, X_group)) : nothing

    ranges = _dmmm_policy_ranges(size(X_group, 1))
    q_chunks = zeros(Float64, length(ranges))
    gradient_alpha_chunks = need_gradient && update_alpha ?
        zeros(Float64, length(ranges), n_components - 1, n_covariates) : nothing
    gradient_phi_chunks = need_gradient && update_phi ?
        zeros(Float64, length(ranges)) : nothing

    @threads :static for chunk in eachindex(ranges)
        q_local = 0.0
        gradient_alpha_local = need_gradient && update_alpha ?
            view(gradient_alpha_chunks, chunk, :, :) : nothing
        gradient_phi_local = 0.0
        for i in ranges[chunk]
            multiplicity = isnothing(row_weights) ? 1.0 : row_weights[i]
            π_i = vec(gate_prob[i, :])
            ξ_i = vec(expected_log_membership[i, :])
            dirichlet_parameter = ϕ .* π_i
            q_local += multiplicity * (
                loggamma(ϕ) - sum(loggamma.(dirichlet_parameter)) +
                sum((dirichlet_parameter .- 1.0) .* ξ_i))

            if need_gradient && update_phi
                gradient_phi_local += multiplicity * (
                    digamma(ϕ) - sum(π_i .* digamma.(dirichlet_parameter)) +
                    sum(π_i .* ξ_i))
            end
            if need_gradient && update_alpha
                a_i = ξ_i .- digamma.(dirichlet_parameter)
                raw_i = vec(raw_gate_prob[i, :])
                if all(>(_DMMM_GATE_FLOOR), raw_i)
                    mean_a = sum(π_i .* a_i)
                    for k in 1:(n_components - 1)
                        gradient_alpha_local[k, :] .+=
                            multiplicity * ϕ * π_i[k] * (a_i[k] - mean_a) .* vec(X_group[i, :])
                    end
                else
                    # The probability floor is piecewise constant. Differentiate
                    # the clipped-and-renormalized gate rather than the raw softmax.
                    clipped = max.(raw_i, _DMMM_GATE_FLOOR)
                    clipped_total = sum(clipped)
                    for k in 1:(n_components - 1)
                        dclipped = [
                            raw_i[j] > _DMMM_GATE_FLOOR ?
                                raw_i[j] * ((j == k) - raw_i[k]) : 0.0
                            for j in 1:n_components
                        ]
                        dtotal = sum(dclipped)
                        dpi = (dclipped .* clipped_total .- clipped .* dtotal) ./
                            clipped_total^2
                        gradient_alpha_local[k, :] .+=
                            multiplicity * ϕ * sum(dpi .* a_i) .* vec(X_group[i, :])
                    end
                end
            end
        end
        q_chunks[chunk] = q_local
        need_gradient && update_phi && (gradient_phi_chunks[chunk] = gradient_phi_local)
    end

    # Reduce fixed chunks in a fixed order so scheduling does not affect results.
    q_value = 0.0
    gradient_alpha = zeros(Float64, n_components - 1, n_covariates)
    gradient_phi = 0.0
    for chunk in eachindex(ranges)
        q_value += q_chunks[chunk]
        need_gradient && update_alpha &&
            (gradient_alpha .+= view(gradient_alpha_chunks, chunk, :, :))
        need_gradient && update_phi &&
            (gradient_phi += gradient_phi_chunks[chunk])
    end

    if penalty
        q_value += penalty_α(α, pen_α)
        update_alpha && (gradient_alpha .-= α[1:(n_components - 1), :] ./ pen_α^2)
    end

    if gradient !== nothing
        update_alpha && (gradient[1:n_alpha] .= -vec(gradient_alpha))
        update_phi && (gradient[end] = -(ϕ * gradient_phi))
    end
    return -q_value
end

function _dmmm_aggregate_gate_profiles(X, expected_log_membership)
    # The Dirichlet objective is linear in xi. Policies with exactly identical
    # X therefore contribute their count and mean xi as sufficient statistics.
    lookup = Dict{Tuple,Int}()
    first_rows = Int[]
    row_profile = Vector{Int}(undef, size(X, 1))
    for i in axes(X, 1)
        profile = Tuple(view(X, i, :))
        index = get(lookup, profile, 0)
        if index == 0
            push!(first_rows, i)
            index = length(first_rows)
            lookup[profile] = index
        end
        row_profile[i] = index
    end
    # Preserve the original summation when there is little or no duplication.
    length(first_rows) > size(X, 1) ÷ 2 && return X, expected_log_membership, nothing
    counts = zeros(Float64, length(first_rows))
    means = zeros(Float64, length(first_rows), size(expected_log_membership, 2))
    for i in axes(X, 1)
        index = row_profile[i]
        counts[index] += 1
        means[index, :] .+= view(expected_log_membership, i, :)
    end
    means ./= counts
    return X[first_rows, :], means, counts
end

function _dmmm_update_dirichlet(
    α,
    ϕ,
    X_group,
    expected_log_membership;
    update_alpha,
    update_phi,
    penalty,
    pen_α,
    dirichlet_iter_max,
    min_ϕ,
    max_ϕ,
)
    (!update_alpha && !update_phi) && return α, ϕ
    X_group, expected_log_membership, row_weights =
        _dmmm_aggregate_gate_profiles(X_group, expected_log_membership)
    n_components, n_covariates = size(α)
    values = Float64[]
    lower = Float64[]
    upper = Float64[]

    if update_alpha
        append!(values, vec(α[1:(n_components - 1), :]))
        append!(lower, fill(-50.0, (n_components - 1) * n_covariates))
        append!(upper, fill(50.0, (n_components - 1) * n_covariates))
    end
    if update_phi
        push!(values, log(ϕ))
        push!(lower, log(min_ϕ))
        push!(upper, log(max_ϕ))
    end
    values .= min.(max.(values, lower .+ 1e-10), upper .- 1e-10)

    objective = value -> _dmmm_neg_q!(
        nothing,
        value,
        α,
        ϕ,
        X_group,
        expected_log_membership;
        update_alpha=update_alpha,
        update_phi=update_phi,
        penalty=penalty,
        pen_α=pen_α,
        row_weights=row_weights,
    )
    gradient! = (storage, value) -> _dmmm_neg_q!(
        storage,
        value,
        α,
        ϕ,
        X_group,
        expected_log_membership;
        update_alpha=update_alpha,
        update_phi=update_phi,
        penalty=penalty,
        pen_α=pen_α,
        row_weights=row_weights,
    )

    result = Optim.optimize(
        objective,
        gradient!,
        lower,
        upper,
        values,
        Optim.Fminbox(Optim.LBFGS()),
        Optim.Options(iterations=dirichlet_iter_max, show_trace=false),
    )
    fitted = Optim.minimizer(result)
    α_new = copy(α)
    cursor = 0
    if update_alpha
        n_alpha = (n_components - 1) * n_covariates
        α_new[1:(n_components - 1), :] .=
            reshape(view(fitted, 1:n_alpha), n_components - 1, n_covariates)
        α_new[end, :] .= 0.0
        cursor = n_alpha
    end
    ϕ_new = update_phi ? exp(fitted[cursor + 1]) : ϕ
    return α_new, ϕ_new
end

function _dmmm_penalty(α, model, penalty, pen_α, pen_params)
    return penalty ? penalty_α(α, pen_α) + penalty_params(model, pen_params) : 0.0
end

function _dmmm_fit_exact(
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
    estep_method,
    max_assignments,
    max_dp_states,
    update_alpha,
    update_phi,
    update_experts,
    zigamma_update_tracker,
    iteration_offset,
    print_steps,
)
    α_em = _dmmm_canonical_alpha(α_init)
    ϕ_em = Float64(ϕ_init)
    model_em = copy(model)
    expert_loglik = _dmmm_expert_logscores(Y, model_em, exposure, response_weights)
    e_step = dmmm_estep(
        expert_loglik,
        X,
        group,
        α_em,
        ϕ_em;
        method=estep_method,
        max_assignments=max_assignments,
        max_dp_states=max_dp_states,
    )
    e_step.method in (:closed_form, :enumeration, :dp) ||
        error("Internal error: exact fitting requires enumeration or DP.")
    ll_np = e_step.ll
    ll = ll_np + _dmmm_penalty(α_em, model_em, penalty, pen_α, pen_params)
    loglik_trace = [ll]
    repeated_groups = any(length(rows) > 1 for rows in e_step.group_index)
    objective_label = _dmmm_is_weighted(response_weights) ? "weighted log criterion" : "loglik"

    update_phi && !repeated_groups &&
        @warn(
            "All policyholders have one observation. The marginal likelihood does not identify ϕ; keeping ϕ at its initial value."
        )

    print_steps > 0 &&
        @info("Initial DMMM $(objective_label): $(ll_np) (no penalty), $(ll) (with penalty)")

    converge = false
    iteration = 0
    for current_iteration in 1:ecm_iter_max
        iteration = current_iteration
        ll_old = ll
        responsibilities = e_step.responsibilities
        expected_log_membership = e_step.expected_log_membership

        if update_experts
            for j in axes(model_em, 1), k in axes(model_em, 2)
                response_weights[j] == 0 && continue
                previous_expert = model_em[j, k]
                candidate = try
                    _em_m_expert_exact_tracked(
                        previous_expert,
                        Y[:, j],
                        exposure,
                        response_weights[j] .* vec(responsibilities[:, k]);
                        penalty=penalty,
                        pen_pararms_jk=pen_params[j][k],
                        tracker=zigamma_update_tracker,
                        diagnostic_key=(j, k),
                        iteration=iteration_offset + current_iteration,
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

                model_em[j, k] = candidate
                candidate_loglik = _dmmm_expert_logscores(
                    Y, model_em, exposure, response_weights
                )
                candidate_e_step = try
                    dmmm_estep(
                        candidate_loglik,
                        X,
                        group,
                        α_em,
                        ϕ_em;
                        method=estep_method,
                        max_assignments=max_assignments,
                        max_dp_states=max_dp_states,
                    )
                catch error
                    nothing
                end
                candidate_ll = if isnothing(candidate_e_step)
                    -Inf
                else
                    candidate_e_step.ll +
                    _dmmm_penalty(α_em, model_em, penalty, pen_α, pen_params)
                end
                if !isfinite(candidate_ll) || candidate_ll < ll - 1e-8
                    model_em[j, k] = previous_expert
                else
                    expert_loglik = candidate_loglik
                    ll_np = candidate_e_step.ll
                    ll = candidate_ll
                end
            end
        end

        update_phi_now = update_phi && repeated_groups
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

        candidate_e_step = dmmm_estep(
            expert_loglik,
            X,
            group,
            α_candidate,
            ϕ_candidate;
            method=estep_method,
            max_assignments=max_assignments,
            max_dp_states=max_dp_states,
        )
        candidate_ll =
            candidate_e_step.ll +
            _dmmm_penalty(α_candidate, model_em, penalty, pen_α, pen_params)
        if isfinite(candidate_ll) && candidate_ll >= ll - 1e-8
            α_em = α_candidate
            ϕ_em = ϕ_candidate
            ll_np = candidate_e_step.ll
            ll = candidate_ll
        end

        expert_loglik = _dmmm_expert_logscores(Y, model_em, exposure, response_weights)
        e_step = dmmm_estep(
            expert_loglik,
            X,
            group,
            α_em,
            ϕ_em;
            method=estep_method,
            max_assignments=max_assignments,
            max_dp_states=max_dp_states,
        )
        ll_np = e_step.ll
        ll = ll_np + _dmmm_penalty(α_em, model_em, penalty, pen_α, pen_params)
        push!(loglik_trace, ll)
        improvement = ll - ll_old

        if print_steps > 0 && current_iteration % print_steps == 0
            @info(
                "DMMM iteration $(current_iteration): $(objective_label) $(ll), increment $(improvement), ϕ $(ϕ_em)"
            )
        end
        if improvement <= ϵ
            converge = improvement >= -1e-8
            break
        end
    end

    if update_phi && repeated_groups &&
       (isapprox(ϕ_em, min_ϕ; rtol=1e-5) || isapprox(ϕ_em, max_ϕ; rtol=1e-5))
        @warn(
            "The fitted ϕ is at an optimization bound; within-policyholder dependence may be weakly identified.",
            ϕ=ϕ_em,
        )
    end

    n_parameters = _count_α(α_em) + _count_params(model_em) + 1
    n_groups = length(e_step.group_ids)
    weighted = _dmmm_is_weighted(response_weights)
    AIC = weighted ? NaN : -2.0 * ll_np + 2.0 * n_parameters
    BIC = weighted ? NaN : -2.0 * ll_np + log(n_groups) * n_parameters
    return (
        α_fit=α_em,
        ϕ_fit=ϕ_em,
        model_fit=model_em,
        converge=converge,
        iter=iteration,
        ll_np=ll_np,
        ll=ll,
        AIC=AIC,
        BIC=BIC,
        group_ids=e_step.group_ids,
        responsibilities=e_step.responsibilities,
        posterior_membership=e_step.posterior_membership,
        loglik_trace=loglik_trace,
        estep_method=e_step.method,
        estep_diagnostics=merge(e_step.diagnostics, (
            zigamma_update_summary=isnothing(zigamma_update_tracker) ? NamedTuple[] :
                zigamma_update_summary(zigamma_update_tracker),)),
        objective_name=weighted ? :weighted_loglik : :loglik,
    )
end
