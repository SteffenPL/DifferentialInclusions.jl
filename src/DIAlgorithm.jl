
abstract type AbstractDIAlgorithm end 

struct ProjectiveMethod{pA, oA} <: AbstractDIAlgorithm 
    proj_alg::pA 
    ode_alg::oA 
end


function SciMLBase.solve(di::DIProblem, alg::ProjectiveMethod; kwargs...)
    each_step(u, t, integrator) = true 
    affect! = alg.proj_alg(di)
    cb = DiscreteCallback(each_step, affect!; save_positions = (false, true))
    return solve(di.ode, alg.ode_alg; save_everystep = false, callback = cb, kwargs...)
end


