"""
    NormalExpert(μ, σ)

A Gaussian expert on the whole real line, parameterized by mean `μ` and
standard deviation `σ`. This expert is suitable for signed responses such as
centered PCA and PLS scores.

The exact-observation EM update is implemented. Exposure does not alter the
distribution.
"""
struct NormalExpert{T<:Real} <: RealContinuousExpert
    μ::T
    σ::T
    NormalExpert{T}(μ::T, σ::T) where {T<:Real} = new{T}(μ, σ)
end

function NormalExpert(μ::T, σ::T; check_args=true) where {T<:Real}
    check_args && @check_args(NormalExpert, σ > zero(σ))
    return NormalExpert{T}(μ, σ)
end

NormalExpert(μ::Real, σ::Real) = NormalExpert(promote(μ, σ)...)
NormalExpert(μ::Integer, σ::Integer) = NormalExpert(float(μ), float(σ))
NormalExpert() = NormalExpert(0.0, 1.0)

function convert(::Type{NormalExpert{T}}, μ::S, σ::S) where {T<:Real,S<:Real}
    return NormalExpert(T(μ), T(σ))
end
function convert(::Type{NormalExpert{T}}, d::NormalExpert{S}) where {T<:Real,S<:Real}
    return NormalExpert(T(d.μ), T(d.σ); check_args=false)
end
copy(d::NormalExpert) = NormalExpert(d.μ, d.σ; check_args=false)

function logpdf(d::NormalExpert, x...)
    return Distributions.logpdf.(Distributions.Normal(d.μ, d.σ), x...)
end
pdf(d::NormalExpert, x...) = Distributions.pdf.(Distributions.Normal(d.μ, d.σ), x...)
function logcdf(d::NormalExpert, x...)
    return Distributions.logcdf.(Distributions.Normal(d.μ, d.σ), x...)
end
cdf(d::NormalExpert, x...) = Distributions.cdf.(Distributions.Normal(d.μ, d.σ), x...)

expert_ll_exact(d::NormalExpert, x::Real) = DMMM.logpdf(d, x)
function expert_ll(d::NormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    result = if yl == yu
        logpdf(d, yl)
    else
        logcdf(d, yu) + log1mexp(logcdf(d, yl) - logcdf(d, yu))
    end
    return tl == tu ? -Inf : result
end

exposurize_expert(d::NormalExpert; exposure=1) = d

params(d::NormalExpert) = (d.μ, d.σ)
function params_init(y, d::NormalExpert)
    observed = collect(skipmissing(y))
    isempty(observed) && return NormalExpert()
    μ_init = mean(observed)
    σ_init = sqrt(var(observed))
    μ_init = isfinite(μ_init) ? μ_init : 0.0
    σ_init = isfinite(σ_init) && σ_init > 0 ? σ_init : 1.0
    return NormalExpert(μ_init, σ_init)
end

function ks_distance(y, d::NormalExpert)
    observed = collect(skipmissing(y))
    return HypothesisTests.ksstats(observed, Distributions.Normal(d.μ, d.σ))[2]
end

sim_expert(d::NormalExpert) = Distributions.rand(Distributions.Normal(d.μ, d.σ), 1)[1]

penalty_init(d::NormalExpert) = [2.0 2.0]
no_penalty_init(d::NormalExpert) = [1.0 1.0]
_normal_variance_prior(p) = p isa NamedTuple ? p.variance : p
function _normal_mean_prior(p)
    eta=p isa NamedTuple ? get(p,:mean_eta,0.) : 0.
    anchor=p isa NamedTuple ? get(p,:mean_variance,1.) : 1.
    isfinite(eta) && eta>=0 && isfinite(anchor) && anchor>0 ||
        throw(ArgumentError("Normal mean penalty requires eta>=0 and a positive finite frozen variance."))
    return Float64(eta),Float64(anchor)
end
function normal_penalty_parts(d::NormalExpert,p)
    h=_normal_variance_prior(p);eta,anchor=_normal_mean_prior(p)
    return (variance=-0.5*(h[1]-1)/d.σ^2-(h[2]-1)*log(d.σ),
        mean=eta==0 ? 0. : -0.5*eta*d.μ^2/anchor)
end
function penalize(d::NormalExpert,p)
    parts=normal_penalty_parts(d,p)
    return parts.variance+parts.mean
end

mean(d::NormalExpert) = d.μ
var(d::NormalExpert) = d.σ^2
quantile(d::NormalExpert, p) = quantile(Distributions.Normal(d.μ, d.σ), p)

function EM_M_expert_exact(
    d::NormalExpert,
    ye,
    exposure,
    z_e_obs;
    penalty=true,
    pen_pararms_jk=[1.0 1.0],
    log_responsibilities=nothing,
)
    observed = .!ismissing.(ye)
    y = Float64.(ye[observed])
    weights = Float64.(z_e_obs[observed])
    supplied=isnothing(log_responsibilities) ? nothing : log_responsibilities[observed]
    lw=_dmmm_responsibility_logs(weights,supplied)
    logR=_dmmm_logweight_sum(lw)
    h=penalty ? _normal_variance_prior(pen_pararms_jk) : [1.,1.]
    eta,anchor=penalty ? _normal_mean_prior(pen_pararms_jk) : (0.,1.)
    if logR==-Inf
        return penalty && h[1]>1 && h[2]>1 ?
            NormalExpert(eta>0 ? 0. : d.μ,sqrt(max(1e-10,(h[1]-1)/(h[2]-1)))) : d
    end
    empirical=_dmmm_signed_weighted_sum(lw.-logR,y)
    logSSE=-Inf
    for i in eachindex(y)
        delta=abs(y[i]-empirical)
        delta>0 && (logSSE=_dmmm_logaddexp(logSSE,lw[i]+2log(delta)))
    end
    return _normal_profile_update(empirical,logR,logSSE,h,eta,anchor)
end

"""Global conditional Normal maximizer, retaining the variance penalty.

For b=sigma², Q=-[(R+h2-1)log b+(SSE+R(mu-ybar)²+h1-1)/b+
eta*mu²/v]/2. The conditional mean is R*ybar/(R+eta*b/v), not
R*ybar/(R+eta/v). Profiling b at fixed mu gives a cubic in z=mu/ybar.
Every stationary root in [0,1] is bracketed between the cubic's turning points
and compared by the actual Q; this handles the possible three-root case.
"""
function _normal_profile_update(m,logR,logSSE,h,eta,anchor;variance_floor=1e-10)
    h[1]>=1 && h[2]>=1 || throw(ArgumentError("Normal optimizer requires h1,h2>=1."))
    logA=_dmmm_logaddexp(logR,log(h[2]-1))
    logC=_dmmm_logaddexp(logSSE,log(h[1]-1))
    eta==0 && return NormalExpert(m,sqrt(max(variance_floor,exp(logC-logA))))
    m==0 && return NormalExpert(0.,sqrt(max(variance_floor,exp(logC-logA))))
    kappa=eta/anchor;A=exp(logA)
    shift(mu)=mu==m ? -Inf : logR+2log(abs(mu-m))
    variance(mu)=exp(_dmmm_logaddexp(logC,shift(mu))-logA)
    q(mu,b)=-.5*(A*log(b)+exp(_dmmm_logaddexp(logC,shift(mu))-log(b))+kappa*mu^2)
    # Cubic: s*z^3-2s*z^2+(u+t+s)*z-u=0, u=R,
    # t=kappa*C/A, s=kappa*R*m²/A. Scale its coefficients, not the likelihood.
    logs=(logR,log(kappa)+logC-logA,log(kappa)+logR+2log(abs(m))-logA)
    scale=maximum(logs);u,t,s=exp.(logs.-scale)
    f(z)=((s*z-2s)*z+(u+t+s))*z-u
    breaks=[0.,1.]
    if s>3(u+t)
        gap=sqrt(max(0.,1-3(u+t)/s))
        append!(breaks,[(2-gap)/3,(2+gap)/3])
    end
    sort!(breaks);candidates=Float64[]
    for i in 1:length(breaks)-1
        left,right=breaks[i],breaks[i+1];fl,fr=f(left),f(right)
        fl==0 && push!(candidates,m*left)
        fr==0 && push!(candidates,m*right)
        (fl!=0 && fr!=0 && signbit(fl)!=signbit(fr)) || continue
        for iteration in 1:1100
            mid=(left+right)/2
            (mid==left || mid==right) && break
            fm=f(mid)
            if fm==0;left=right=mid;break;end
            if signbit(fm)==signbit(fl);left=mid;fl=fm;else;right=mid;end
        end
        push!(candidates,m*((left+right)/2))
    end
    # A broad numerical variance floor is a constrained boundary candidate.
    # Mean is still its exact conditional maximizer; no PC-mean bound is used.
    floor_mu=sign(m)*exp(logR+log(abs(m))-_dmmm_logaddexp(logR,log(kappa*variance_floor)))
    choices=[(mu,variance(mu)) for mu in candidates if variance(mu)>=variance_floor]
    push!(choices,(floor_mu,variance_floor))
    # When a scaled coefficient rounds to zero, retain the limiting conditional
    # mean evaluated in log scale at the prior-dominated variance.
    if u==0
        b=max(variance_floor,exp(logC-logA))
        mu=sign(m)*exp(logR+log(abs(m))-_dmmm_logaddexp(logR,log(kappa)+log(b)))
        push!(choices,(mu,b))
    end
    objective=[q(mu,b) for (mu,b) in choices]
    best=choices[argmax(objective)]
    isfinite(best[1]) && isfinite(best[2]) || throw(ArgumentError("Nonfinite Normal candidate."))
    return NormalExpert(best[1],sqrt(best[2]))
end
