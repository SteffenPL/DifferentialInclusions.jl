
abstract type AbstractDIAlgorithm end 

struct ProjectiveMethod{pA, oA} <: AbstractDIAlgorithm 
    proj_alg::pA 
    ode_alg::oA 
end


struct PseudoSymmetricProjectiveMethod{pA, oA} <: AbstractDIAlgorithm 
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
    return solve(get_ode(di, alg)...; save_everystep = false, kwargs...)
end


function get_ode(di::DIProblem, alg::ProjectiveMethod; kwarfs...)
    each_step(u, t, integrator) = true 
    non_smooth_int = alg.proj_alg(di)
    cb = DiscreteCallback(each_step, non_smooth_int; save_positions = (false, true))

    prob = remake(di.ode, callback = cb)
    return prob, alg.ode_alg
end

function SciMLBase.solve(di::DIProblem, alg::PenaltyMethod; kwargs...)
    return solve(get_ode(di, alg)...; kwargs...)
end


function get_ode(di::DIProblem, alg::PenaltyMethod; kwarfs...)
    di_int = alg(di) 
    return di_int.mod_ode, alg.ode_alg
end
