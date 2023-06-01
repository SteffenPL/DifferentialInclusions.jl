
abstract type AbstractDIAlgorithm end 

struct ProjectiveMethod{pA, oA} <: AbstractDIAlgorithm 
    proj_alg::pA 
    ode_alg::oA 
end

Base.@kwdef struct PenaltyMethod{oA} <: AbstractDIAlgorithm
    ode_alg::oA
    alpha::Float64 = 2.0
    gamma::Float64 = 1e2
end

PenaltyMethod(ode_alg; kwargs...) = PenaltyMethod(;ode_alg, kwargs...)

# Fallback: Solve only the ODE! (Useful for penalty methods)
function SciMLBase.solve(di::DIProblem, alg; kwargs...)
    return solve(di.ode,alg; kwargs...)
end

function SciMLBase.solve(di::DIProblem, alg::ProjectiveMethod; kwargs...)
    each_step(u, t, integrator) = true 
    non_smooth_int = alg.proj_alg(di)
    cb = DiscreteCallback(each_step, non_smooth_int; save_positions = (false, true))

    return solve(di.ode, alg.ode_alg; save_everystep = false, callback = cb, kwargs...)
end


function SciMLBase.solve(di::DIProblem, alg::PenaltyMethod; kwargs...)
    di_int = alg(di) 
    return solve(di_int.mod_ode, alg.ode_alg; kwargs...)
end
