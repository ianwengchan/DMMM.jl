struct ZIGammaExpert{T<:Real} <: ZIContinuousExpert
    p::T
    k::T
    θ::T
    ZIGammaExpert{T}(p::T, k::T, θ::T) where {T<:Real} = new{T}(p, k, θ)
end

"""Mutable, caller-owned trajectory diagnostics for boundary-safe ZIGamma updates."""
mutable struct ZIGammaUpdateRecord
    opportunities::Int
    fitted::Int
    skipped::Int
    first_fitted::Union{Nothing,Int}
    last_fitted::Union{Nothing,Int}
    first_skipped::Union{Nothing,Int}
    last_skipped::Union{Nothing,Int}
    final_status::Symbol
    resumed_after_skip::Bool
    skip_reasons::Dict{Symbol,Int}
end

ZIGammaUpdateRecord() = ZIGammaUpdateRecord(
    0, 0, 0, nothing, nothing, nothing, nothing, :none, false, Dict{Symbol,Int}())

mutable struct ZIGammaUpdateTracker
    records::Dict{Tuple{Int,Int},ZIGammaUpdateRecord}
end
ZIGammaUpdateTracker() = ZIGammaUpdateTracker(Dict{Tuple{Int,Int},ZIGammaUpdateRecord}())

function _record_zigamma_update!(tracker, key, iteration, status, reason=nothing)
    isnothing(tracker) && return
    isnothing(key) && throw(ArgumentError("diagnostic_key is required with a ZIGamma tracker."))
    record = get!(tracker.records, Tuple(key), ZIGammaUpdateRecord())
    record.opportunities += 1
    if status == :fitted
        record.fitted += 1
        record.first_fitted = isnothing(record.first_fitted) ? iteration : record.first_fitted
        record.last_fitted = iteration
        record.resumed_after_skip |= record.skipped > 0
    else
        record.skipped += 1
        record.first_skipped = isnothing(record.first_skipped) ? iteration : record.first_skipped
        record.last_skipped = iteration
        record.skip_reasons[reason] = get(record.skip_reasons, reason, 0) + 1
    end
    record.final_status = status
    return
end

function zigamma_update_summary(tracker::ZIGammaUpdateTracker)
    return [(
        response_dimension=key[1], component=key[2],
        opportunities=r.opportunities, fitted=r.fitted, skipped=r.skipped,
        fraction_skipped=r.opportunities == 0 ? 0.0 : r.skipped / r.opportunities,
        first_fitted=r.first_fitted, last_fitted=r.last_fitted,
        first_skipped=r.first_skipped, last_skipped=r.last_skipped,
        final_status=r.final_status, resumed_after_skip=r.resumed_after_skip,
        skip_reasons=copy(r.skip_reasons),
    ) for (key, r) in sort!(collect(tracker.records); by=first)]
end

function ZIGammaExpert(p::T, k::T, θ::T; check_args=true) where {T<:Real}
    check_args && @check_args(ZIGammaExpert, 0 <= p <= 1 && k >= zero(k) && θ > zero(θ))
    return ZIGammaExpert{T}(p, k, θ)
end

#### Outer constructors
ZIGammaExpert(p::Real, k::Real, θ::Real) = ZIGammaExpert(promote(p, k, θ)...)
function ZIGammaExpert(p::Integer, k::Integer, θ::Integer)
    return ZIGammaExpert(float(p), float(k), float(θ))
end
ZIGammaExpert() = ZIGammaExpert(0.50, 1.0, 1.0)

## Conversion
function convert(::Type{ZIGammaExpert{T}}, p::S, k::S, θ::S) where {T<:Real,S<:Real}
    return ZIGammaExpert(T(p), T(k), T(θ))
end
function convert(::Type{ZIGammaExpert{T}}, d::ZIGammaExpert{S}) where {T<:Real,S<:Real}
    return ZIGammaExpert(T(d.p), T(d.k), T(d.θ); check_args=false)
end
copy(d::ZIGammaExpert) = ZIGammaExpert(d.p, d.k, d.θ; check_args=false)

## Loglikelihood of Expoert
function logpdf(d::ZIGammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        -Inf
    else
        Distributions.logpdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end
function pdf(d::ZIGammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        0.0
    else
        Distributions.pdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end
function logcdf(d::ZIGammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        -Inf
    else
        Distributions.logcdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end
function cdf(d::ZIGammaExpert, x...)
    return if (d.k < 1 && x... <= 0.0)
        0.0
    else
        Distributions.cdf.(Distributions.Gamma(d.k, d.θ), x...)
    end
end

## expert_ll, etc
function expert_ll_exact(d::ZIGammaExpert, x::Real)
    return (x == 0.0) ? log(p_zero(d)) : log(1 - p_zero(d)) + DMMM.logpdf(d, x)
end
function expert_ll(d::ZIGammaExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = DMMM.expert_ll(DMMM.GammaExpert(d.k, d.θ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = if (yl == 0.0)
        log.(p0 + (1 - p0) * exp.(expert_ll_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_ll_pos))
    end
    expert_ll = (tu == 0.0) ? log.(p0) : expert_ll
    return expert_ll
end
function expert_tn(d::ZIGammaExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = DMMM.expert_tn(DMMM.GammaExpert(d.k, d.θ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn = if (tl == 0.0)
        log.(p0 + (1 - p0) * exp.(expert_tn_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_tn_pos))
    end
    expert_tn = (tu == 0.0) ? log.(p0) : expert_tn
    return expert_tn
end
function expert_tn_bar(d::ZIGammaExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = DMMM.expert_tn_bar(DMMM.GammaExpert(d.k, d.θ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn_bar = if (tl > 0.0)
        log.(p0 + (1 - p0) * exp.(expert_tn_bar_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_tn_bar_pos))
    end
    return expert_tn_bar
end

exposurize_expert(d::ZIGammaExpert; exposure=1) = d

## Parameters
params(d::ZIGammaExpert) = (d.p, d.k, d.θ)
p_zero(d::ZIGammaExpert) = d.p
function params_init(y, d::ZIGammaExpert)
    p_init = sum(y .== 0.0) / sum(y .>= 0.0)
    pos_idx = (y .> 0.0)
    μ, σ2 = mean(y[pos_idx]), var(y[pos_idx])
    θ_init = σ2 / μ
    k_init = μ / θ_init
    if isnan(θ_init) || isnan(k_init)
        return ZIGammaExpert()
    else
        return ZIGammaExpert(p_init, k_init, θ_init)
    end
end

## KS stats for parameter initialization
function ks_distance(y, d::ZIGammaExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(
        abs(p_zero - d.p),
        (1 - d.p) * HypothesisTests.ksstats(y[y .> 0.0], Distributions.Gamma(d.k, d.θ))[2],
    )
end

## Simululation
function sim_expert(d::ZIGammaExpert)
    return (1 .- Distributions.rand(Distributions.Bernoulli(d.p), 1)[1]) .*
           Distributions.rand(Distributions.Gamma(d.k, d.θ), 1)[1]
end

## penalty
penalty_init(d::ZIGammaExpert) = [2.0 10.0 2.0 10.0]
no_penalty_init(d::ZIGammaExpert) = [1.0 Inf 1.0 Inf]
function penalize(d::ZIGammaExpert, p)
    return (p[1] - 1) * log(d.k) - d.k / p[2] + (p[3] - 1) * log(d.θ) - d.θ / p[4]
end

## statistics
mean(d::ZIGammaExpert) = (1 - d.p) * mean(Distributions.Gamma(d.k, d.θ))
function var(d::ZIGammaExpert)
    return (1 - d.p) * var(Distributions.Gamma(d.k, d.θ)) +
           d.p * (1 - d.p) * (mean(Distributions.Gamma(d.k, d.θ)))^2
end
function quantile(d::ZIGammaExpert, p)
    return p <= d.p ? 0.0 : quantile(Distributions.Gamma(d.k, d.θ), p - d.p)
end
lev(d::ZIGammaExpert, u) = (1 - d.p) * lev(GammaExpert(d.k, d.θ), u)
excess(d::ZIGammaExpert, u) = mean(d) - lev(d, u)

## EM: M-Step
function EM_M_expert(d::ZIGammaExpert,
    tl, yl, yu, tu,
    exposure,
    #  expert_ll_pos,
    #  expert_tn_pos,
    #  expert_tn_bar_pos,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[1.0 Inf 1.0 Inf],
    positive_support_min=2.0, update_diagnostics=nothing,
    diagnostic_key=nothing, iteration=0)

    # Old parameters
    p_old = p_zero(d)

    positive_support_min >= 0 || throw(ArgumentError("positive_support_min must be nonnegative."))

    # Update zero probability
    expert_ll_pos = expert_ll.(DMMM.GammaExpert(d.k, d.θ), tl, yl, yu, tu)
    expert_tn_bar_pos = expert_tn_bar.(DMMM.GammaExpert(d.k, d.θ), tl, yl, yu, tu)

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(yl, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = z_e_lat .* EM_E_z_zero_lat(tl, p_old, expert_tn_bar_pos)
    z_pos_e_lat = z_e_lat .- z_zero_e_lat
    total_support = sum(z_zero_e_obs .+ z_pos_e_obs) +
        sum((z_zero_e_lat .+ z_pos_e_lat) .* k_e)
    p_new = isfinite(total_support) && total_support > 0 ?
        EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e) : p_old

    s_positive = sum(z_pos_e_obs) + sum(z_pos_e_lat .* k_e)
    if !isfinite(s_positive)
        _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration,
            :skipped, :nonfinite_positive_statistics)
        return ZIGammaExpert(p_new, d.k, d.θ)
    elseif s_positive < positive_support_min
        _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration,
            :skipped, :insufficient_positive_support)
        return ZIGammaExpert(p_new, d.k, d.θ)
    end

    # For uncensored positive observations, detect the same zero-variance
    # boundary before entering the ordinary Gamma sufficient-statistic update.
    exact_positive = (yu .> 0) .& (yl .== yu)
    if all((z_pos_e_obs .== 0) .| exact_positive) && sum(z_pos_e_obs) > 0
        values = yl[exact_positive]
        weights = z_pos_e_obs[exact_positive]
        support = sum(weights)
        weighted_mean = sum(weights .* values) / support
        weighted_variance = sum(weights .* (values .- weighted_mean) .^ 2) / support
        if !(isfinite(weighted_mean) && isfinite(weighted_variance) && weighted_mean > 0)
            _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration,
                :skipped, :nonfinite_positive_statistics)
            return ZIGammaExpert(p_new, d.k, d.θ)
        elseif weighted_variance <= eps(Float64) * max(weighted_mean^2, 1.0)
            _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration,
                :skipped, :degenerate_positive_variance)
            return ZIGammaExpert(p_new, d.k, d.θ)
        end
    end

    # Update parameters: call its positive part
    tmp_exp = GammaExpert(d.k, d.θ)
    tmp_update = EM_M_expert(tmp_exp,
        tl, yl, yu, tu,
        exposure,
        # expert_ll_pos,
        # expert_tn_pos,
        # expert_tn_bar_pos,
        # z_e_obs, z_e_lat, k_e,
        z_pos_e_obs, z_pos_e_lat, k_e;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk)

    if !(isfinite(tmp_update.k) && isfinite(tmp_update.θ) &&
         tmp_update.k > 0 && tmp_update.θ > 0)
        _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration,
            :skipped, :other_numerical_failure)
        error("Supported ZIGamma positive-part update produced invalid Gamma parameters.")
    end
    _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration, :fitted)

    return ZIGammaExpert(p_new, tmp_update.k, tmp_update.θ)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZIGammaExpert,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[Inf 1.0 Inf],
    positive_support_min=2.0, update_diagnostics=nothing,
    diagnostic_key=nothing, iteration=0)

    # Old parameters
    positive_support_min >= 0 || throw(ArgumentError("positive_support_min must be nonnegative."))

    # With exact observations, zero/positive membership is observed.  This
    # direct split remains well-defined even when the previous p is on a boundary.
    zero_idx = ye .== 0.0
    pos_idx = ye .> 0.0
    z_zero_e_obs = ifelse.(zero_idx, z_e_obs, zero(eltype(z_e_obs)))
    z_pos_e_obs = ifelse.(pos_idx, z_e_obs, zero(eltype(z_e_obs)))
    total_support = sum(z_zero_e_obs) + sum(z_pos_e_obs)
    p_new = isfinite(total_support) && total_support > 0 ?
        EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0) : p_zero(d)

    s_positive = sum(z_pos_e_obs)
    if !isfinite(s_positive)
        _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration,
            :skipped, :nonfinite_positive_statistics)
        return ZIGammaExpert(p_new, d.k, d.θ)
    elseif s_positive < positive_support_min
        _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration,
            :skipped, :insufficient_positive_support)
        return ZIGammaExpert(p_new, d.k, d.θ)
    end

    positive_y = ye[pos_idx]
    positive_w = z_pos_e_obs[pos_idx]
    weighted_mean = sum(positive_w .* positive_y) / s_positive
    weighted_variance = sum(positive_w .* (positive_y .- weighted_mean) .^ 2) / s_positive
    if !(isfinite(weighted_mean) && isfinite(weighted_variance) && weighted_mean > 0)
        _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration,
            :skipped, :nonfinite_positive_statistics)
        return ZIGammaExpert(p_new, d.k, d.θ)
    end
    variance_tolerance = eps(Float64) * max(weighted_mean^2, 1.0)
    if weighted_variance <= variance_tolerance
        _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration,
            :skipped, :degenerate_positive_variance)
        return ZIGammaExpert(p_new, d.k, d.θ)
    end

    # Update parameters: call its positive part
    tmp_exp = GammaExpert(d.k, d.θ)
    tmp_update = EM_M_expert_exact(tmp_exp,
        ye, exposure,
        z_pos_e_obs;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk)

    if !(isfinite(tmp_update.k) && isfinite(tmp_update.θ) &&
         tmp_update.k > 0 && tmp_update.θ > 0)
        _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration,
            :skipped, :other_numerical_failure)
        error("Supported ZIGamma positive-part update produced invalid Gamma parameters.")
    end

    _record_zigamma_update!(update_diagnostics, diagnostic_key, iteration, :fitted)

    return ZIGammaExpert(p_new, tmp_update.k, tmp_update.θ)
end

function _em_m_expert_exact_tracked(d, ye, exposure, weights;
    penalty, pen_pararms_jk, tracker=nothing, diagnostic_key=nothing, iteration=0)
    return EM_M_expert_exact(d, ye, exposure, weights;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk)
end

function _em_m_expert_exact_tracked(d::ZIGammaExpert, ye, exposure, weights;
    penalty, pen_pararms_jk, tracker=nothing, diagnostic_key=nothing, iteration=0)
    return EM_M_expert_exact(d, ye, exposure, weights;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk,
        update_diagnostics=tracker, diagnostic_key=diagnostic_key, iteration=iteration)
end
