# Exact one-dimensional observed criterion for panels of at most three periods.
# Cost is O(N K) to build coefficients and O(N) per phi evaluation. This never
# computes an exact E-step and is independent of the emission distribution.
function _dmmm_phi_terms(scores, X_group, group_index, alpha)
    all(rows -> 1 <= length(rows) <= 3, group_index) || throw(ArgumentError(
        "MCEM phi profiles currently require one to three observations per group."))
    gates = _dmmm_gate_probabilities(alpha, X_group)
    terms = fill(-Inf, length(group_index), 3)
    periods = length.(group_index)
    ranges = _dmmm_policy_ranges(length(group_index))
    @threads :static for chunk in eachindex(ranges)
        for i in ranges[chunk]
            rows = group_index[i]
            g = log.(view(gates, i, :))
            u = view(scores, rows[1], :)
            m1 = logsumexp(g .+ u)
            if periods[i] == 1
                terms[i, 1] = m1
                continue
            end
            v = view(scores, rows[2], :)
            m2, d12 = logsumexp(g .+ v), logsumexp(g .+ u .+ v)
            if periods[i] == 2
                terms[i, 1:2] .= (d12, m1 + m2)
                continue
            end
            w = view(scores, rows[3], :)
            m3 = logsumexp(g .+ w)
            terms[i, 1] = log(2.0) + logsumexp(g .+ u .+ v .+ w)
            terms[i, 2] = logsumexp([d12 + m3,
                logsumexp(g .+ u .+ w) + m2, logsumexp(g .+ v .+ w) + m1])
            terms[i, 3] = m1 + m2 + m3
        end
    end
    all(isfinite, maximum(terms; dims=2)) || throw(ArgumentError("Non-finite phi profile."))
    return (coefficients=terms, periods=periods)
end

function _dmmm_phi_value(terms, phi)
    lp = log(phi)
    ranges = _dmmm_policy_ranges(length(terms.periods))
    value_terms = zeros(Float64, length(terms.periods))
    score_terms = zeros(Float64, length(terms.periods))
    @threads :static for chunk in eachindex(ranges)
        for i in ranges[chunk]
            a = terms.coefficients[i, 1]
            b = terms.coefficients[i, 2] + lp
            c = terms.coefficients[i, 3] + 2lp
            anchor = max(a, b, c)
            ea, eb, ec = exp(a-anchor), exp(b-anchor), exp(c-anchor)
            total = ea + eb + ec
            value_terms[i] = anchor + log(total)
            score_terms[i] = (eb + 2ec)/total
        end
    end
    # Replay the original policy-order reduction exactly; only the expensive
    # independent per-policy terms above are evaluated in parallel.
    value = 0.0
    score = 0.0
    for i in eachindex(terms.periods)
        value += value_terms[i]
        score += score_terms[i]
        for t in 1:(terms.periods[i]-1)
            value -= log(phi+t)
            score -= phi/(phi+t)
        end
    end
    return (value=value, score=score)
end

function _dmmm_phi_profile(terms, phi; min_phi=1e-6, max_phi=1e6, grid_size=81)
    grid = sort!(unique!(vcat(collect(range(log(min_phi), log(max_phi); length=grid_size)), log(phi))))
    evaluated = [_dmmm_phi_value(terms, exp(x)) for x in grid]
    candidates = [phi, min_phi, max_phi]
    # Refine every sampled local maximum bracket, not just the first root.
    for i in 1:(length(grid)-1)
        if evaluated[i].score > 0 && evaluated[i+1].score <= 0
            lo, hi = grid[i], grid[i+1]
            for unused in 1:50
                mid = (lo+hi)/2
                if _dmmm_phi_value(terms, exp(mid)).score > 0
                    lo = mid
                else
                    hi = mid
                end
            end
            push!(candidates, exp((lo+hi)/2))
        end
    end
    values = [_dmmm_phi_value(terms, p).value for p in candidates]
    best = argmax(values)
    current = _dmmm_phi_value(terms, phi)
    flat = maximum(x.value for x in evaluated)-minimum(x.value for x in evaluated) < 1e-7
    # A flat profile provides no data-informed initialization.
    optimum = flat ? phi : candidates[best]
    return (phi_current=phi, phi_profile=optimum, criterion=current.value,
        criterion_gain=flat ? 0.0 : max(0.0, values[best]-current.value),
        log_phi_score=current.score, log_phi_distance=abs(log(optimum/phi)),
        at_bound=optimum == min_phi || optimum == max_phi, flat=flat)
end

function _dmmm_phi_controls(options)
    options isa NamedTuple || throw(ArgumentError("mcem_phi_gate must be nothing or a NamedTuple."))
    defaults = (window=10, check_interval=10, gain_tol_per_group=1e-4,
        phi_stability_tol=0.05,
        credibility_T=2.0, credibility_tol=0.01,
        criterion_tol_per_row=1e-4, gate_tv_tol=0.01,
        expert_min_support=10.0, expert_atol=1e-4, expert_rtol=0.05,
        stable_checks=1, mc_replicates=3, mc_log_phi_sd_tol=0.01,
        mc_diagnostic=true, grid_size=81, rescue=:none,
        rescue_log_distance=0.75, rescue_gain_per_group=1e-3,
        rescue_windows=2, rescue_rho=0.25, max_rescues=4)
    all(k -> k in keys(defaults), keys(options)) || throw(ArgumentError("Unknown mcem_phi_gate option."))
    controls = merge(defaults, options)
    for k in (:window, :check_interval, :stable_checks, :mc_replicates,
              :grid_size, :rescue_windows, :max_rescues)
        v = getproperty(controls, k)
        minimum = k == :max_rescues ? 0 : k == :mc_replicates ? 2 : k == :grid_size ? 5 : 1
        v isa Integer && v >= minimum || throw(ArgumentError("Invalid phi gate control $k."))
    end
    for k in (:gain_tol_per_group, :phi_stability_tol,
              :credibility_T, :credibility_tol, :criterion_tol_per_row,
              :gate_tv_tol, :expert_min_support, :expert_atol, :expert_rtol,
              :mc_log_phi_sd_tol, :rescue_log_distance,
              :rescue_gain_per_group)
        v = getproperty(controls, k)
        v isa Real && isfinite(v) && v >= 0 || throw(ArgumentError("Invalid phi gate tolerance $k."))
    end
    controls.credibility_T > 0 || throw(ArgumentError("credibility_T must be positive."))
    controls.mc_diagnostic isa Bool || throw(ArgumentError("mc_diagnostic must be Bool."))
    controls.rescue in (:none, :damped) ||
        throw(ArgumentError("phi rescue must be :none or :damped."))
    isfinite(controls.rescue_rho) && 0 < controls.rescue_rho <= 1 ||
        throw(ArgumentError("rescue_rho must be in (0, 1]."))
    return controls
end

function _dmmm_parameter_snapshot(model)
    values, components = Float64[], Int[]
    for k in axes(model, 2), j in axes(model, 1)
        p = params(model[j, k])
        parameters = p isa Number ? (p,) : p
        append!(values, parameters)
        append!(components, fill(k, length(parameters)))
    end
    return (values=values, components=components)
end

function _dmmm_phi_stability(history, controls, X_group, nrows)
    length(history) >= controls.window+1 || return (stable=false, phi_stable=false,
        phi_change=Inf, criterion_change_per_row=Inf, gate_tv=Inf,
        expert_scaled_change=Inf, supported_components=0)
    first, last = history[1], history[end]
    # Require the whole recent window to be stable, not just its endpoints.
    phi_change = maximum(abs(log(s.phi/last.phi)) for s in history)
    criterion_change = maximum(abs(s.criterion-last.criterion) for s in history)/nrows
    supported = [maximum(s.support[k] for s in history) >= controls.expert_min_support
                 for k in eachindex(last.support)]
    selected = [supported[k] for k in last.parameter_components]
    expert_change = any(selected) ? maximum(maximum((abs.(s.parameters-last.parameters) ./
        (controls.expert_atol .+ controls.expert_rtol .* abs.(last.parameters)))[selected])
        for s in history) : Inf
    latest_gates = _dmmm_gate_probabilities(last.alpha, X_group)
    gate_tv = maximum(sum(abs.(_dmmm_gate_probabilities(s.alpha, X_group)-latest_gates)) /
        (2size(X_group, 1)) for s in history)
    phi_stable = phi_change <= controls.phi_stability_tol
    stable = phi_stable && criterion_change <= controls.criterion_tol_per_row &&
        gate_tv <= controls.gate_tv_tol && expert_change <= 1.0
    return (stable=stable, phi_stable=phi_stable, phi_change=phi_change,
        criterion_change_per_row=criterion_change,
        gate_tv=gate_tv, expert_scaled_change=expert_change,
        supported_components=count(supported))
end
