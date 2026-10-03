module DMMM

import Base: size, length, convert, show, getindex, rand, vec, inv, expm1, abs, log1p
import Base: isnan, isinf
import Base: sum, maximum, minimum, ceil, floor, extrema, +, -, *, ==
import Base: convert, copy, findfirst, summary
import Base.Math: @horner
import Base: π
import Base.Threads: @threads, nthreads, threadid

using StatsFuns
import StatsFuns: log1mexp, log1pexp, logsumexp
import StatsFuns: sqrt2, invsqrt2π

using Statistics
import Statistics: quantile, mean, var, median

using Distributions
import Distributions: pdf, cdf, ccdf, logpdf, logcdf, logccdf, quantile
import Distributions: rand, AbstractRNG
import Distributions: mean, var, skewness, kurtosis
import Distributions:
    UnivariateDistribution, DiscreteUnivariateDistribution, ContinuousUnivariateDistribution
import Distributions: @distr_support, RecursiveProbabilityEvaluator
import Distributions: Bernoulli, Multinomial
import Distributions: Binomial, Poisson
import Distributions: Gamma, InverseGaussian, LogNormal, Normal, Weibull

using InvertedIndices
import InvertedIndices: Not

using LinearAlgebra
import LinearAlgebra: I, Cholesky

using SpecialFunctions
import SpecialFunctions: erf, loggamma, gamma_inc, gamma, beta_inc

using QuadGK
import QuadGK: quadgk

using Optim
import Optim: optimize, minimizer

using Clustering
import Clustering: kmeans, assignments, counts

using HypothesisTests
import HypothesisTests: ExactOneSampleKSTest, pvalue, ksstats

using Logging

using Random

using Roots
import Roots: find_zero, Order2

export
    DMMMModel,
    DMMMFit,
    fit_DMMM,
    dmmm_estep,
    dmmm_expert_logscores,
    dmmm_logcriterion,
    estimate_phi_moments,
    simulate_dmmm,
    predict_class_prior,
    predict_class_posterior_dmmm,
    predict_mean_posterior_dmmm,
    predict_var_posterior_dmmm,
    predictive_mixture_dmmm,
    DMMMPredictiveMixture,
    BurrExpert,
    ZIBurrExpert,
    GammaExpert,
    ZIGammaExpert,
    InverseGaussianExpert,
    ZIInverseGaussianExpert,
    LogNormalExpert,
    ZILogNormalExpert,
    NormalExpert,
    ZOIBetaExpert,
    WeibullExpert,
    ZIWeibullExpert,
    BinomialExpert,
    ZIBinomialExpert,
    GammaCountExpert,
    ZIGammaCountExpert,
    NegativeBinomialExpert,
    ZINegativeBinomialExpert,
    PoissonExpert,
    ZIPoissonExpert

### source files

include("utils.jl")
include("AICBIC.jl")
include("modelstruct.jl")

include("gating.jl")
include("expert.jl")
include("loglik.jl")
include("penalty.jl")

include("paramsinit.jl")
include("fit.jl")

include("simulation.jl")
include("predict.jl")

"""
Julia implementation of the Dirichlet Mixed-Membership Model (DMMM).
"""
DMMM

end # module
