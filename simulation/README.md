# Simulation machinery

The package exports `simulate_dmmm`, a generic in-memory generator for the
Dirichlet Mixed-Membership Model. Callers supply the baseline covariates,
policyholder/history identifiers, gate coefficients, total Dirichlet
concentration, class-specific experts, exposures, and a Julia random-number
generator. The result can include the simulated responses, period classes, and
fixed policyholder compositions.

See `examples/minimal_fit.jl` for a complete synthetic workflow.

The research workspace also contains a study-specific Python donor-matching
application. It is not included here because it is coupled to a particular
external dataframe schema, telematics variables, study margins, and output
manifests. No proprietary source data or generated pseudo-panels are included.

