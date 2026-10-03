"""
    estimate_phi_moments(H, component_means, X, group, α;
                         weights=:equal, denominator_tolerance=sqrt(eps(Float64)))

Estimate the DMMM dependence parameters by the two-stage method of moments.
`H` is the observation-by-transformation matrix whose rows are `h(Y_it)`,
and `component_means[:, k]` is `E[h(Y_it) | C_it = k]`. The marginal LRMoE
parameters `α` and the component means are treated as fixed first-stage
estimates.

For each group with at least two observations, the estimator constructs the
average symmetrized within-group cross-product `C_i` and its DMMM target
`ρ B_i`, then performs the scalar Frobenius least-squares regression

`ρ̃ = sum_i w_i <C_i, B_i>_F / sum_i w_i ||B_i||_F^2`.

The estimate is projected to `[0, 1]` and transformed using
`ϕ = (1 - ρ) / ρ`. A projected estimate `ρ = 0` therefore returns `ϕ = Inf`.
Use `weights=:equal` to weight groups equally or `weights=:pairs` to weight
each within-group observation pair equally. A nonnegative numeric vector with
one entry per group may also be supplied.

`X` may contain one row per observation, provided it is constant within group,
or one row per group in first-appearance order.
"""
function estimate_phi_moments(
    H,
    component_means,
    X,
    group,
    α;
    weights=:equal,
    denominator_tolerance=sqrt(eps(Float64)),
)
    H_array = Matrix{Float64}(H)
    component_mean_array = Matrix{Float64}(component_means)
    X_array = Matrix{Float64}(X)
    α_array = Matrix{Float64}(α)
    group_array = collect(group)

    n_observations, n_features = size(H_array)
    n_observations >= 1 || throw(ArgumentError("H must contain at least one row."))
    n_features >= 1 || throw(ArgumentError("H must contain at least one column."))
    length(group_array) == n_observations ||
        throw(DimensionMismatch("group must contain one identifier per row of H."))
    size(component_mean_array, 1) == n_features ||
        throw(
            DimensionMismatch(
                "component_means must have one row per transformed-response feature."
            ),
        )
    size(component_mean_array, 2) == size(α_array, 1) ||
        throw(
            DimensionMismatch(
                "component_means and α must contain the same number of components."
            ),
        )
    all(isfinite, H_array) || throw(ArgumentError("H must contain only finite values."))
    all(isfinite, component_mean_array) ||
        throw(ArgumentError("component_means must contain only finite values."))
    all(isfinite, X_array) || throw(ArgumentError("X must contain only finite values."))
    all(isfinite, α_array) || throw(ArgumentError("α must contain only finite values."))
    isfinite(denominator_tolerance) && denominator_tolerance >= 0 ||
        throw(
            ArgumentError(
                "denominator_tolerance must be finite and nonnegative."
            ),
        )

    group_ids, group_index, unused_row_group = _dmmm_groups(group_array)
    X_group = _dmmm_group_covariates(X_array, group_index, n_observations)
    size(X_group, 2) == size(α_array, 2) ||
        throw(DimensionMismatch("The numbers of covariates in X and α must agree."))
    gate_prob = _dmmm_gate_probabilities(α_array, X_group)

    group_weights = if weights === :equal
        ones(Float64, length(group_ids))
    elseif weights === :pairs
        [
            length(rows) * (length(rows) - 1) / 2 for rows in group_index
        ]
    elseif weights isa AbstractVector
        length(weights) == length(group_ids) ||
            throw(
                DimensionMismatch(
                    "A numeric weights vector must contain one entry per group."
                ),
            )
        Float64.(weights)
    else
        throw(
            ArgumentError(
                "weights must be :equal, :pairs, or a nonnegative numeric vector."
            ),
        )
    end
    all(weight -> isfinite(weight) && weight >= 0, group_weights) ||
        throw(ArgumentError("All group weights must be finite and nonnegative."))

    numerator = 0.0
    denominator = 0.0
    used_group_ids = Any[]
    used_group_weights = Float64[]
    group_cross_products = Matrix{Float64}[]
    group_targets = Matrix{Float64}[]

    for i in eachindex(group_index)
        rows = group_index[i]
        n_periods = length(rows)
        n_periods >= 2 || continue
        weight = group_weights[i]
        weight > 0 || continue

        π_i = vec(gate_prob[i, :])
        marginal_mean = component_mean_array * π_i
        centered = H_array[rows, :] .- marginal_mean'
        centered_sum = vec(sum(centered; dims=1))

        # This identity evaluates the average symmetrized pair cross-product in
        # O(T_i q^2), avoiding explicit enumeration of T_i(T_i-1)/2 pairs.
        cross_product =
            (
                centered_sum * centered_sum' -
                Matrix(centered' * centered)
            ) / (n_periods * (n_periods - 1))

        target = zeros(Float64, n_features, n_features)
        for k in axes(component_mean_array, 2)
            difference = vec(component_mean_array[:, k]) .- marginal_mean
            target .+= π_i[k] .* (difference * difference')
        end

        numerator += weight * dot(cross_product, target)
        denominator += weight * sum(abs2, target)
        push!(used_group_ids, group_ids[i])
        push!(used_group_weights, weight)
        push!(group_cross_products, cross_product)
        push!(group_targets, target)
    end

    isempty(used_group_ids) &&
        throw(
            ArgumentError(
                "At least one positive-weight group with two observations is required."
            ),
        )
    denominator_scale =
        sum(used_group_weights) *
        max(sum(abs2, component_mean_array), 1.0)
    denominator > denominator_tolerance * denominator_scale ||
        throw(
            ArgumentError(
                "The moment denominator is numerically zero. The chosen transformation " *
                "does not sufficiently distinguish the fitted expert means."
            ),
        )

    rho_unprojected = numerator / denominator
    rho = clamp(rho_unprojected, 0.0, 1.0)
    phi = iszero(rho) ? Inf : (1 - rho) / rho
    return (
        phi=phi,
        rho=rho,
        rho_unprojected=rho_unprojected,
        numerator=numerator,
        denominator=denominator,
        n_groups=length(used_group_ids),
        group_ids=used_group_ids,
        group_weights=used_group_weights,
        group_cross_products=group_cross_products,
        group_targets=group_targets,
        gate_prob=gate_prob,
        weights=weights,
    )
end
