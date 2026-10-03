using DMMM
using Distributions
using Random
using Statistics
using Test

@testset "DMMM package load and model basics" begin
    alpha = [-0.4 0.2; 0.0 0.0]
    X = [1.0 -0.5; 1.0 0.8]
    pi = predict_class_prior(X, alpha).prob
    @test size(pi) == (2, 2)
    @test vec(sum(pi; dims=2)) ≈ ones(2)

    phi = 3.5
    for i in axes(pi, 1)
        law = Dirichlet(phi .* vec(pi[i, :]))
        @test mean(law) ≈ vec(pi[i, :])
        @test sum(mean(law)) ≈ 1.0
        @test all(>(0), law.alpha)
    end
end

@testset "exact E-step agreement" begin
    alpha = [-0.35 0.20; 0.10 -0.15; 0.0 0.0]
    group_X = [1.0 -0.5; 1.0 0.8]
    group = [1, 1, 1, 2, 2]
    X = group_X[[1, 1, 1, 2, 2], :]
    density = [
        0.65 0.25 0.10
        0.15 0.70 0.15
        0.40 0.20 0.40
        0.20 0.30 0.50
        0.55 0.35 0.10
    ]
    phi = 2.75

    enumeration = dmmm_estep(log.(density), X, group, alpha, phi; method=:enumeration)
    grouping = dmmm_estep(log.(density), X, group, alpha, phi; method=:closed_form)
    dp = dmmm_estep(log.(density), X, group, alpha, phi; method=:dp)
    for exact in (grouping, dp)
        @test exact.ll ≈ enumeration.ll atol=1e-12
        @test exact.group_loglik ≈ enumeration.group_loglik atol=1e-12
        @test exact.responsibilities ≈ enumeration.responsibilities atol=1e-12
        @test exact.expected_log_membership ≈ enumeration.expected_log_membership atol=1e-12
        @test exact.posterior_membership ≈ enumeration.posterior_membership atol=1e-12
    end

    weighted_scores = log.(density) .+ [0.2, -0.4, 0.6, -0.1, 0.3]
    weighted_enumeration = dmmm_estep(
        weighted_scores, X, group, alpha, phi; method=:enumeration
    )
    weighted_grouping = dmmm_estep(
        weighted_scores, X, group, alpha, phi; method=:closed_form
    )
    weighted_dp = dmmm_estep(weighted_scores, X, group, alpha, phi; method=:dp)
    for exact in (weighted_grouping, weighted_dp)
        @test exact.ll ≈ weighted_enumeration.ll atol=1e-12
        @test exact.responsibilities ≈ weighted_enumeration.responsibilities atol=1e-12
        @test exact.expected_log_membership ≈
              weighted_enumeration.expected_log_membership atol=1e-12
    end
end

@testset "unit response weights recover ordinary scores" begin
    Y = [0.0 0.2; 1.0 -0.1; 4.0 1.8; 3.0 2.2]
    model = Any[
        PoissonExpert(1.0) PoissonExpert(4.0)
        NormalExpert(0.0, 1.0) NormalExpert(2.0, 1.0)
    ]
    ordinary = dmmm_expert_logscores(Y, model)
    unit_weighted = dmmm_expert_logscores(Y, model; response_weights=[1.0, 1.0])
    @test isequal(ordinary, unit_weighted)

    downweighted = dmmm_expert_logscores(Y, model; response_weights=[1.0, 0.25])
    @test all(isfinite, downweighted)
    @test downweighted != ordinary

    y = [0.0, 1.0, 3.0, 5.0]
    responsibilities = [0.7, 0.8, 0.4, 0.2]
    unit_fit = DMMM.EM_M_expert_exact(
        PoissonExpert(2.0), y, ones(4), responsibilities;
        penalty=false, pen_pararms_jk=[1.0 Inf],
    )
    scaled_fit = DMMM.EM_M_expert_exact(
        PoissonExpert(2.0), y, ones(4), 4.0 .* responsibilities;
        penalty=false, pen_pararms_jk=[1.0 Inf],
    )
    @test unit_fit.λ ≈ scaled_fit.λ atol=1e-12
end

@testset "clipped gate gradient follows the stabilized objective" begin
    alpha = zeros(2, 1)
    X = ones(1, 1)
    xi = reshape([-1.5, -0.4], 1, 2)
    function objective(value)
        DMMM._dmmm_neg_q!(
            nothing, [value], alpha, 2.0, X, xi;
            update_alpha=true, update_phi=false, penalty=false, pen_α=1.0,
        )
    end
    gradient = zeros(1)
    DMMM._dmmm_neg_q!(
        gradient, [-40.0], alpha, 2.0, X, xi;
        update_alpha=true, update_phi=false, penalty=false, pen_α=1.0,
    )
    step = 1e-4
    finite_difference = (objective(-40.0 + step) - objective(-40.0 - step)) / (2step)
    @test gradient[1] ≈ finite_difference atol=1e-9
end

@testset "seeded simulation, small fit, and predictive mixture" begin
    alpha = reshape([log(0.55 / 0.45), 0.0], 2, 1)
    phi = 3.0
    group = repeat(1:8; inner=2)
    X = ones(length(group), 1)
    true_model = reshape(Any[PoissonExpert(1.0), PoissonExpert(5.0)], 1, 2)

    first = simulate_dmmm(
        alpha, phi, X, group, true_model;
        rng=MersenneTwister(20261003), return_latent=true,
    )
    second = simulate_dmmm(
        alpha, phi, X, group, true_model;
        rng=MersenneTwister(20261003), return_latent=true,
    )
    @test first == second
    @test size(first.Y) == (length(group), 1)
    @test size(first.membership) == (8, 2)
    @test vec(sum(first.membership; dims=2)) ≈ ones(8)

    initial_model = reshape(Any[PoissonExpert(1.2), PoissonExpert(4.5)], 1, 2)
    fit = fit_DMMM(
        first.Y,
        X,
        group,
        zeros(2, 1),
        2.0,
        initial_model;
        estep=:closed_form,
        exact_update=:block,
        penalty=false,
        ecm_iter_max=2,
        ϵ=0.0,
        print_steps=0,
    )
    @test isfinite(fit.loglik)
    @test fit.model_fit.ϕ > 0
    @test fit.model_fit.α[end, :] == zeros(size(fit.model_fit.α, 2))
    @test size(fit.responsibilities) == (length(group), 2)
    @test vec(sum(fit.responsibilities; dims=2)) ≈ ones(length(group))
    @test vec(sum(fit.posterior_membership; dims=2)) ≈ ones(8)

    predicted_mean = predict_mean_posterior_dmmm(first.Y, X, group, fit)
    @test size(predicted_mean) == (8, 1)
    @test all(isfinite, predicted_mean)

    mixtures = predictive_mixture_dmmm(first.Y, X, group, fit)
    mixture = mixtures[1, 1]
    @test sum(pdf(mixture, value) for value in 0:80) ≈ 1.0 atol=1e-10
    @test mean(mixture) ≈ predicted_mean[1, 1] atol=1e-12
    @test isfinite(var(mixture))

    zero_inflated = DMMMPredictiveMixture(
        [0.4, 0.6], [ZIPoissonExpert(0.25, 1.0), ZIPoissonExpert(0.50, 4.0)]
    )
    @test sum(pdf(zero_inflated, value) for value in 0:100) ≈ 1.0 atol=1e-12
    @test cdf(zero_inflated, 100) ≈ 1.0 atol=1e-12
end
