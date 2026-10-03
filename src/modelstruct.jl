# Abstract type: whether the model has random effects
abstract type RandomEffects end
struct hasRandomEffects <: RandomEffects end
struct noRandomEffects <: RandomEffects end

# Model
abstract type LRMoEModel{re<:RandomEffects} end

const LRMoEModelSTD = LRMoEModel{noRandomEffects}
const LRMoEModelRE = LRMoEModel{hasRandomEffects}

# Result
abstract type LRMoEFittingResult end

# Model contains: regression coefficients, component distributions
#   number of iteration, convergence, likelihood, AIC, BIC -> Move to another struct
struct LRMoESTD <: LRMoEModelSTD
    α::Array
    comp_dist::Array
    function LRMoESTD(α, comp_dist)
        return (
            if size(α)[1] == size(comp_dist)[2]
                new(α, comp_dist)
            else
                error("Invalid specification of model.")
            end
        )
    end
end

struct LRMoESTDFit <: LRMoEFittingResult
    model_fit::LRMoESTD
    converge::Bool
    iter::Integer
    loglik::Real
    loglik_np::Real
    AIC::Real
    BIC::Real
    objective_name::Symbol
    response_weights::Vector{Float64}
end

# Backward-compatible constructor for ordinary, unweighted LRMoE fits.
LRMoESTDFit(model_fit, converge, iter, loglik, loglik_np, AIC, BIC) =
    LRMoESTDFit(
        model_fit, converge, iter, loglik, loglik_np, AIC, BIC,
        :loglik, ones(Float64, size(model_fit.comp_dist, 1)),
    )

"""
    DMMMModel(α, ϕ, comp_dist)

A fitted Dirichlet Mixed-Membership Model. `α` contains the logit-gating
coefficients, `ϕ` is the positive Dirichlet concentration, and `comp_dist`
contains the expert distributions.
"""
struct DMMMModel <: LRMoEModelRE
    α::Array
    ϕ::Real
    comp_dist::Array
    function DMMMModel(α, ϕ, comp_dist)
        size(α, 1) == size(comp_dist, 2) ||
            error("The number of gating classes and expert components must agree.")
        isfinite(ϕ) && ϕ > 0 ||
            error("The Dirichlet concentration ϕ must be finite and positive.")
        return new(α, ϕ, comp_dist)
    end
end

"""
    DMMMFit

Result returned by [`fit_DMMM`](@ref). `responsibilities` contains the
period-specific posterior class probabilities. `posterior_membership` contains
the posterior mean of the fixed policyholder membership vector, one row per
entry of `group_ids`. `estep_method` records the selected backend. For MCEM,
`loglik`, `loglik_np`, `AIC`, and `BIC` are `NaN`, while `loglik_trace` stores
the Monte Carlo EM `Q` objective and `objective_name == :Q`.
`response_weights` records the fixed response powers used for estimation and
by default for subsequent history updating. Non-unit weights change the objective
name to `:weighted_loglik` or `:weighted_Q` and disable ordinary AIC/BIC.
For exact weighted fitting, `loglik`/`loglik_np` contain the weighted criterion
(with/without penalties), not an ordinary log density.
"""
struct DMMMFit <: LRMoEFittingResult
    model_fit::DMMMModel
    converge::Bool
    iter::Integer
    loglik::Real
    loglik_np::Real
    AIC::Real
    BIC::Real
    group_ids::Vector
    responsibilities::Matrix{Float64}
    posterior_membership::Matrix{Float64}
    loglik_trace::Vector{Float64}
    estep_method::Symbol
    estep_diagnostics::NamedTuple
    objective_name::Symbol
    response_weights::Vector{Float64}
end

# Compatibility with result objects constructed before response weighting.
DMMMFit(
    model_fit, converge, iter, loglik, loglik_np, AIC, BIC, group_ids,
    responsibilities, posterior_membership, loglik_trace, estep_method,
    estep_diagnostics, objective_name,
) = DMMMFit(
    model_fit, converge, iter, loglik, loglik_np, AIC, BIC, group_ids,
    responsibilities, posterior_membership, loglik_trace, estep_method,
    estep_diagnostics, objective_name, ones(Float64, size(model_fit.comp_dist, 1)),
)

# Backward-compatible constructor for code that created result objects directly
# before E-step metadata was added.
DMMMFit(
    model_fit,
    converge,
    iter,
    loglik,
    loglik_np,
    AIC,
    BIC,
    group_ids,
    responsibilities,
    posterior_membership,
    loglik_trace,
) = DMMMFit(
    model_fit,
    converge,
    iter,
    loglik,
    loglik_np,
    AIC,
    BIC,
    group_ids,
    responsibilities,
    posterior_membership,
    loglik_trace,
    :enumeration,
    NamedTuple(),
    :loglik,
)

"""
    summary(obj)

Summarizes a fitted LRMoE model.

# Arguments
- `obj`: An object returned by `fit_LRMoE` function.

# Return Values
Prints out a summary of the fitted DMMM model on screen.
"""
function summary(m::LRMoESTDFit)
    println("Model: LRMoE")
    if m.converge
        println("Fitting converged after $(m.iter) iterations")
    else
        println("Fitting NOT converged after $(m.iter) iterations")
    end
    println("Dimension of response: $(size(m.model_fit.comp_dist)[1])")
    println("Number of components: $(size(m.model_fit.comp_dist)[2])")
    println("Response weights: $(m.response_weights)")
    if _dmmm_is_weighted(m.response_weights)
        println("Objective: $(m.objective_name) (weighted composite criterion)")
        println("Objective value: $(m.loglik)")
        println("Objective value (no penalty): $(m.loglik_np)")
        println("Ordinary AIC and BIC: not applicable to non-unit response weights")
    else
        println("Loglik: $(m.loglik)")
        println("Loglik (no penalty): $(m.loglik_np)")
        println("AIC: $(m.AIC)")
        println("BIC: $(m.BIC)")
    end
    println("Fitted α:")
    println("$(m.model_fit.α)")
    println("Fitted component distributions:")
    return println("$(m.model_fit.comp_dist)")
end

function summary(m::DMMMFit)
    println("Model: Dirichlet Mixed-Membership Model (DMMM)")
    if m.converge
        println("Fitting converged after $(m.iter) iterations")
    else
        println("Fitting NOT converged after $(m.iter) iterations")
    end
    println("Number of policyholders: $(length(m.group_ids))")
    println("Dimension of response: $(size(m.model_fit.comp_dist, 1))")
    println("Number of components: $(size(m.model_fit.comp_dist, 2))")
    println("Dirichlet concentration ϕ: $(m.model_fit.ϕ)")
    println("E-step backend: $(m.estep_method)")
    println("Response weights: $(m.response_weights)")
    println(
        "Latent-class intrapolicyholder correlation 1/(ϕ+1): $(
            1 / (m.model_fit.ϕ + 1)
        )"
    )
    if _dmmm_is_weighted(m.response_weights)
        println("Objective: $(m.objective_name) (weighted composite criterion)")
        !isempty(m.loglik_trace) && println("Final objective: $(last(m.loglik_trace))")
        println("Ordinary AIC and BIC: not applicable to non-unit response weights")
    elseif m.estep_method == :mcem
        println("Observed loglik, AIC, and BIC: unavailable for MCEM")
        !isempty(m.loglik_trace) &&
            println("Final Monte Carlo EM Q: $(last(m.loglik_trace))")
    else
        println("Loglik: $(m.loglik)")
        println("Loglik (no penalty): $(m.loglik_np)")
        println("AIC: $(m.AIC)")
        println("BIC: $(m.BIC)")
    end
    println("Fitted α:")
    println("$(m.model_fit.α)")
    println("Fitted component distributions:")
    return println("$(m.model_fit.comp_dist)")
end
