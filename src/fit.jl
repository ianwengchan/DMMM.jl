const fit = [
    "fit_main",
    "em",
    "fit_exact",
    "dmmm_weights",
    "dmmm_moments",
    "dmmm_exact",
    "dmmm_phi_profile",
    "dmmm_estep",
    "fit_interface",
]

for dname in fit
    include(joinpath("fit", "$(dname).jl"))
end
