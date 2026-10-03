# Response weights alter emissions only, never the Dirichlet/categorical layer.
function _dmmm_response_weights(response_weights, n_responses)
    isnothing(response_weights) && return ones(Float64, n_responses)
    response_weights isa AbstractVector ||
        throw(ArgumentError("response_weights must be a vector or nothing."))
    length(response_weights) == n_responses ||
        throw(DimensionMismatch("response_weights must have one entry per response."))
    all(w -> w isa Real && isfinite(w) && w >= 0, response_weights) ||
        throw(ArgumentError("response_weights must be finite and nonnegative."))
    weights = Float64.(response_weights)
    all(isfinite, weights) ||
        throw(ArgumentError("response_weights must be representable as finite Float64 values."))
    return weights
end

_dmmm_is_weighted(weights) = any(!=(1.0), weights)

function _dmmm_expert_logscores(Y, model, exposure, weights)
    # Preserve the original numerical path when all weights are one.
    !_dmmm_is_weighted(weights) &&
        return _exact_expert_ll_threaded(Y, model, exposure)
    result = zeros(Float64, size(Y, 1), size(model, 2))
    @threads for row in axes(Y, 1)
        for j in axes(model, 1)
            weights[j] == 0 && continue # Avoid 0 * -Inf, and do not evaluate omitted factors.
            for k in axes(model, 2)
                expert = exposurize_expert(model[j, k]; exposure=exposure[row])
                result[row, k] += weights[j] * expert_ll_exact(expert, Y[row, j])
            end
        end
    end
    return result
end

"""
    dmmm_expert_logscores(Y, model; exposure=nothing, response_weights=nothing)

Evaluate the observation-by-class scores `sum(w[r] * log(f[r,k](Y[r])))`
from ver3 equation (31), without fitting. Weights are fixed, nonnegative,
response-specific, and NOT normalized. `nothing` means all ones. Zero-weight
factors are omitted, even if their density at the observed value is zero.
The result can be passed to any `dmmm_estep` backend.
"""
function dmmm_expert_logscores(Y, model; exposure=nothing, response_weights=nothing)
    Y_array = Array(Y)
    ndims(Y_array) == 2 && ndims(model) == 2 ||
        throw(ArgumentError("Y and model must be matrices."))
    size(Y_array, 2) == size(model, 1) ||
        throw(DimensionMismatch("Y and model must have the same response dimensions."))
    all(isfinite, Y_array) ||
        throw(ArgumentError("Y must contain only finite exact outcomes."))
    weights = _dmmm_response_weights(response_weights, size(model, 1))
    exposure = isnothing(exposure) ? ones(size(Y_array, 1)) : collect(exposure)
    length(exposure) == size(Y_array, 1) ||
        throw(DimensionMismatch("exposure must contain one value per observation."))
    all(e -> isfinite(e) && e > 0, exposure) ||
        throw(ArgumentError("exposure values must be finite and positive."))
    return _dmmm_expert_logscores(Y_array, model, exposure, weights)
end

"""
    dmmm_logcriterion(Y, X, group, α, ϕ, model;
                       exposure=nothing, response_weights=nothing,
                       method=:enumeration, max_assignments=1_000_000,
                       max_dp_states=2_000_000)

Evaluate the exact integrated log criterion (ver3 equations (33)-(34)) at fixed
parameters, without fitting or penalties. Returns `value`, `group_values`,
`criterion_name`, `response_weights`, and `estep_result`. With non-unit weights
this is a weighted composite criterion, NOT an ordinary predictive log density.
Only enumeration and DP are supported here; MCEM has no exact log criterion.
"""
function dmmm_logcriterion(
    Y, X, group, α, ϕ, model;
    exposure=nothing, response_weights=nothing, method=:enumeration,
    max_assignments=1_000_000, max_dp_states=2_000_000,
)
    method in (:enumeration, :dp) ||
        throw(ArgumentError("Exact criterion evaluation requires :enumeration or :dp."))
    weights = _dmmm_response_weights(response_weights, size(model, 1))
    scores = dmmm_expert_logscores(
        Y, model; exposure=exposure, response_weights=weights
    )
    result = dmmm_estep(
        scores, X, group, α, ϕ; method=method,
        max_assignments=max_assignments, max_dp_states=max_dp_states,
    )
    return (
        value=result.ll, group_values=result.group_loglik,
        criterion_name=_dmmm_is_weighted(weights) ? :weighted_loglik : :loglik,
        response_weights=weights, estep_result=result,
    )
end
