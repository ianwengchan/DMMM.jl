using DMMM
using Random

rng = MersenneTwister(2026)
group = repeat(1:20; inner=3)
X = ones(length(group), 1)
alpha = reshape([log(0.6 / 0.4), 0.0], 2, 1)
experts = reshape(Any[PoissonExpert(1.0), PoissonExpert(5.0)], 1, 2)

simulated = simulate_dmmm(
    alpha,
    3.0,
    X,
    group,
    experts;
    rng=rng,
    return_latent=true,
)

initial_experts = reshape(Any[PoissonExpert(1.3), PoissonExpert(4.0)], 1, 2)
fit = fit_DMMM(
    simulated.Y,
    X,
    group,
    zeros(2, 1),
    2.0,
    initial_experts;
    estep=:closed_form,
    exact_update=:block,
    penalty=false,
    print_steps=0,
)

posterior = predict_class_posterior_dmmm(simulated.Y, X, group, fit)
future_mean = predict_mean_posterior_dmmm(simulated.Y, X, group, fit)
future_distribution = predictive_mixture_dmmm(simulated.Y, X, group, fit)

println("Posterior composition for the first policyholder: ", posterior.prob[1, :])
println("Predictive mean for the first policyholder: ", future_mean[1, :])
println("Predictive mixture object: ", future_distribution[1, 1])

