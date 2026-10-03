"""
    ZOIBetaExpert(p0, p1, α, β)

Zero-one-inflated Beta expert on `[0, 1]`:

```math
P(Y=0)=p_0,\\quad P(Y=1)=p_1,\\quad
Y\\mid 0<Y<1 \\sim \\operatorname{Beta}(\\alpha,\\beta).
```

The continuous component has weight `1 - p0 - p1`. This expert is intended
for proportions observed exactly. Its exact-observation M-step is implemented;
the interval-censored M-step is deliberately unsupported.
"""
struct ZOIBetaExpert{T<:Real} <: ZIContinuousExpert
    p0::T
    p1::T
    α::T
    β::T
    function ZOIBetaExpert{T}(p0::T, p1::T, α::T, β::T) where {T<:Real}
        return new{T}(p0, p1, α, β)
    end
end

function ZOIBetaExpert(
    p0::T, p1::T, α::T, β::T; check_args=true
) where {T<:Real}
    check_args && @check_args(
        ZOIBetaExpert,
        zero(p0) <= p0 <= one(p0) &&
            zero(p1) <= p1 <= one(p1) &&
            p0 + p1 <= one(p0) &&
            α > zero(α) &&
            β > zero(β),
    )
    return ZOIBetaExpert{T}(p0, p1, α, β)
end

ZOIBetaExpert(p0::Real, p1::Real, α::Real, β::Real) =
    ZOIBetaExpert(promote(p0, p1, α, β)...)
ZOIBetaExpert(p0::Integer, p1::Integer, α::Integer, β::Integer) =
    ZOIBetaExpert(float(p0), float(p1), float(α), float(β))
ZOIBetaExpert() = ZOIBetaExpert(0.05, 0.05, 2.0, 2.0)

function convert(
    ::Type{ZOIBetaExpert{T}}, p0::S, p1::S, α::S, β::S
) where {T<:Real,S<:Real}
    return ZOIBetaExpert(T(p0), T(p1), T(α), T(β))
end
function convert(
    ::Type{ZOIBetaExpert{T}}, d::ZOIBetaExpert{S}
) where {T<:Real,S<:Real}
    return ZOIBetaExpert(
        T(d.p0), T(d.p1), T(d.α), T(d.β); check_args=false
    )
end
copy(d::ZOIBetaExpert) =
    ZOIBetaExpert(d.p0, d.p1, d.α, d.β; check_args=false)

_zoibeta_component(d::ZOIBetaExpert) = Distributions.Beta(d.α, d.β)
_zoibeta_continuous_probability(d::ZOIBetaExpert) = 1 - d.p0 - d.p1
_zoibeta_log_probability(p::Real) = p <= 0 ? -Inf : log(p)

# As for the package's other inflated experts, these methods describe the
# continuous component. Mixture mass is incorporated by expert_ll_exact.
logpdf(d::ZOIBetaExpert, x::Real) =
    Distributions.logpdf(_zoibeta_component(d), x)
pdf(d::ZOIBetaExpert, x::Real) =
    Distributions.pdf(_zoibeta_component(d), x)
logcdf(d::ZOIBetaExpert, x::Real) =
    Distributions.logcdf(_zoibeta_component(d), x)
cdf(d::ZOIBetaExpert, x::Real) =
    Distributions.cdf(_zoibeta_component(d), x)

function expert_ll_exact(d::ZOIBetaExpert, x::Real)
    if x == 0
        return _zoibeta_log_probability(d.p0)
    elseif x == 1
        return _zoibeta_log_probability(d.p1)
    elseif 0 < x < 1
        return _zoibeta_log_probability(
            _zoibeta_continuous_probability(d)
        ) + DMMM.logpdf(d, x)
    end
    return -Inf
end

function _zoibeta_interval_probability(
    d::ZOIBetaExpert, lower::Real, upper::Real
)
    upper < lower && return 0.0
    probability = 0.0
    lower <= 0 <= upper && (probability += d.p0)
    lower <= 1 <= upper && (probability += d.p1)

    beta_lower = clamp(float(lower), 0.0, 1.0)
    beta_upper = clamp(float(upper), 0.0, 1.0)
    if beta_upper > beta_lower
        component_probability =
            Distributions.cdf(_zoibeta_component(d), beta_upper) -
            Distributions.cdf(_zoibeta_component(d), beta_lower)
        probability +=
            _zoibeta_continuous_probability(d) *
            max(component_probability, 0.0)
    end
    return clamp(probability, 0.0, 1.0)
end

function expert_ll(
    d::ZOIBetaExpert, tl::Real, yl::Real, yu::Real, tu::Real
)
    yl == yu && return expert_ll_exact(d, yl)
    return _zoibeta_log_probability(
        _zoibeta_interval_probability(d, yl, yu)
    )
end

function expert_tn(
    d::ZOIBetaExpert, tl::Real, yl::Real, yu::Real, tu::Real
)
    tl == tu && return expert_ll_exact(d, tl)
    return _zoibeta_log_probability(
        _zoibeta_interval_probability(d, tl, tu)
    )
end

function expert_tn_bar(
    d::ZOIBetaExpert, tl::Real, yl::Real, yu::Real, tu::Real
)
    retained = if tl == tu
        tl == 0 ? d.p0 : (tl == 1 ? d.p1 : 0.0)
    else
        _zoibeta_interval_probability(d, tl, tu)
    end
    return _zoibeta_log_probability(max(1 - retained, 0.0))
end

exposurize_expert(d::ZOIBetaExpert; exposure=1) = d

params(d::ZOIBetaExpert) = (d.p0, d.p1, d.α, d.β)
p_zero(d::ZOIBetaExpert) = d.p0
p_one(d::ZOIBetaExpert) = d.p1

function _zoibeta_moment_shapes(y, weights=nothing)
    isempty(y) && return (2.0, 2.0)
    w = weights === nothing ? ones(Float64, length(y)) : Float64.(vec(weights))
    total_weight = sum(w)
    total_weight > 0 || return (2.0, 2.0)
    μ = sum(w .* y) / total_weight
    σ2 = sum(w .* (y .- μ) .^ 2) / total_weight
    if !(0 < μ < 1) || !isfinite(σ2) || σ2 <= eps(Float64)
        return (2.0, 2.0)
    end
    concentration = μ * (1 - μ) / σ2 - 1
    if !isfinite(concentration) || concentration <= 0
        concentration = 10.0
    end
    return (
        max(μ * concentration, 1e-3),
        max((1 - μ) * concentration, 1e-3),
    )
end

function params_init(y, d::ZOIBetaExpert)
    values = Float64.(vec(y))
    any(value -> !isfinite(value) || value < 0 || value > 1, values) &&
        return ZOIBetaExpert()
    n = length(values)
    n == 0 && return ZOIBetaExpert()
    p0 = count(==(0.0), values) / n
    p1 = count(==(1.0), values) / n
    interior = values[(values .> 0) .& (values .< 1)]
    α, β = _zoibeta_moment_shapes(interior)
    return ZOIBetaExpert(p0, p1, α, β)
end

function ks_distance(y, d::ZOIBetaExpert)
    values = Float64.(vec(y))
    valid = values[(values .>= 0) .& (values .<= 1)]
    isempty(valid) && return Inf
    empirical_p0 = count(==(0.0), valid) / length(valid)
    empirical_p1 = count(==(1.0), valid) / length(valid)
    interior = valid[(valid .> 0) .& (valid .< 1)]
    continuous_distance = isempty(interior) ? 0.0 :
        HypothesisTests.ksstats(interior, _zoibeta_component(d))[2]
    return maximum([
        abs(empirical_p0 - d.p0),
        abs(empirical_p1 - d.p1),
        _zoibeta_continuous_probability(d) * continuous_distance,
    ])
end

function sim_expert(d::ZOIBetaExpert)
    draw = Distributions.rand(Distributions.Uniform())
    draw <= d.p0 && return 0.0
    draw >= 1 - d.p1 && return 1.0
    return Distributions.rand(_zoibeta_component(d))
end

penalty_init(d::ZOIBetaExpert) = [2.0 10.0 2.0 10.0]
no_penalty_init(d::ZOIBetaExpert) = [1.0 Inf 1.0 Inf]
function penalize(d::ZOIBetaExpert, p)
    return (p[1] - 1) * log(d.α) - d.α / p[2] +
           (p[3] - 1) * log(d.β) - d.β / p[4]
end

function mean(d::ZOIBetaExpert)
    component_mean = mean(_zoibeta_component(d))
    return d.p1 + _zoibeta_continuous_probability(d) * component_mean
end

function var(d::ZOIBetaExpert)
    component = _zoibeta_component(d)
    component_second_moment = var(component) + mean(component)^2
    second_moment =
        d.p1 +
        _zoibeta_continuous_probability(d) * component_second_moment
    return max(second_moment - mean(d)^2, 0.0)
end

function quantile(d::ZOIBetaExpert, probability)
    0 <= probability <= 1 ||
        throw(ArgumentError("quantile probability must lie in [0, 1]."))
    probability <= d.p0 && return 0.0
    probability >= 1 - d.p1 && return 1.0
    continuous_probability = _zoibeta_continuous_probability(d)
    continuous_probability > 0 || return 1.0
    adjusted = (probability - d.p0) / continuous_probability
    return quantile(_zoibeta_component(d), adjusted)
end

function lev(d::ZOIBetaExpert, u)
    u <= 0 && return 0.0
    u >= 1 && return mean(d)
    component = _zoibeta_component(d)
    partial_first_moment =
        mean(component) *
        Distributions.cdf(Distributions.Beta(d.α + 1, d.β), u)
    limited_component =
        partial_first_moment +
        u * (1 - Distributions.cdf(component, u))
    return d.p1 * u +
           _zoibeta_continuous_probability(d) * limited_component
end

excess(d::ZOIBetaExpert, u) = mean(d) - lev(d, u)

function EM_M_expert(
    d::ZOIBetaExpert,
    tl,
    yl,
    yu,
    tu,
    exposure,
    z_e_obs,
    z_e_lat,
    k_e;
    penalty=true,
    pen_pararms_jk=[1.0 Inf 1.0 Inf],
)
    throw(ArgumentError(
        "ZOIBetaExpert currently supports exact observations only; " *
        "use EM_M_expert_exact.",
    ))
end

function _zoibeta_shape_objective(
    log_shapes,
    y,
    weights;
    penalty=true,
    pen_pararms_jk=[1.0 Inf 1.0 Inf],
)
    α = exp(log_shapes[1])
    β = exp(log_shapes[2])
    if !isfinite(α) || !isfinite(β)
        return Inf
    end
    total_weight = sum(weights)
    log_normalizer = loggamma(α) + loggamma(β) - loggamma(α + β)
    objective =
        (α - 1) * sum(weights .* log.(y)) +
        (β - 1) * sum(weights .* log1p.(-y)) -
        total_weight * log_normalizer
    if penalty
        objective +=
            (pen_pararms_jk[1] - 1) * log(α) -
            α / pen_pararms_jk[2] +
            (pen_pararms_jk[3] - 1) * log(β) -
            β / pen_pararms_jk[4]
    end
    result = -objective
    return isfinite(result) ? result : Inf
end

function EM_M_expert_exact(
    d::ZOIBetaExpert,
    ye,
    exposure,
    z_e_obs;
    penalty=true,
    pen_pararms_jk=[1.0 Inf 1.0 Inf],
)
    values = Float64.(vec(ye))
    weights = Float64.(vec(z_e_obs))
    length(values) == length(weights) ||
        throw(DimensionMismatch("ye and z_e_obs must have equal lengths."))
    any(value -> !isfinite(value) || value < 0 || value > 1, values) &&
        throw(ArgumentError("ZOIBetaExpert observations must lie in [0, 1]."))
    any(weight -> !isfinite(weight) || weight < 0, weights) &&
        throw(ArgumentError("ZOIBetaExpert weights must be finite and nonnegative."))

    total_weight = sum(weights)
    total_weight > 0 || return d
    zero_index = values .== 0
    one_index = values .== 1
    interior_index = (values .> 0) .& (values .< 1)
    p0 = sum(weights[zero_index]) / total_weight
    p1 = sum(weights[one_index]) / total_weight

    interior_values = values[interior_index]
    interior_weights = weights[interior_index]
    if sum(interior_weights) <= eps(Float64) ||
       length(unique(interior_values)) < 2
        return ZOIBetaExpert(p0, p1, d.α, d.β)
    end

    moment_α, moment_β =
        _zoibeta_moment_shapes(interior_values, interior_weights)
    initial = [
        log(isfinite(moment_α) ? moment_α : d.α),
        log(isfinite(moment_β) ? moment_β : d.β),
    ]
    result = Optim.optimize(
        shapes -> _zoibeta_shape_objective(
            shapes,
            interior_values,
            interior_weights;
            penalty=penalty,
            pen_pararms_jk=pen_pararms_jk,
        ),
        initial,
    )
    shapes = Optim.minimizer(result)
    α = exp(shapes[1])
    β = exp(shapes[2])
    if !isfinite(α) || !isfinite(β) || α <= 0 || β <= 0
        α, β = d.α, d.β
    end
    return ZOIBetaExpert(p0, p1, α, β)
end
