using ForwardDiff
using OrdinaryDiffEq
using LinearAlgebra


##############
## Setup    ##
##############

struct DIProblem{odeT,consT}
    ode::odeT 
    cons::consT 
end

struct OneStepPJ
end

struct OneStepPJIntegrator{consT}
    cons::consT
end

(::OneStepPJ)(di::DIProblem) = OneStepPJIntegrator(di.cons)

function newton_raphson!(u, con) 
    c = con(u) 
    if c < 0 
        dc = ForwardDiff.gradient(con, u)
        Δu = -c / dot(dc, dc) * dc
        u .+= Δu
    end
    return nothing 
end 

function (cs::OneStepPJIntegrator)(integrator)
    foreach( con -> newton_raphson!(integrator.u, con), cs.cons)
end


function SciMLBase.solve(di::DIProblem, alg; kwargs...)

    each_step(u, t, integrator) = true 
    affect! = alg.lincompl(di)
    cb = DiscreteCallback(each_step, affect!; save_positions = (false, true))

    return solve(di.ode, alg.ode; save_everystep = false, callback = cb, kwargs...)
end

##############
##  Example ##
##############
cons = (
    u -> u[1], 
    u -> 2.0 - u[2], 
    u -> abs(u[1] - u[2]) - 1.0
)

function ode!(du, u, p, t)
    du[1] = -p.gamma * u[1]
    du[2] = -p.gamma * u[2]
end

ode = ODEProblem( ode!, [0.0, 2.0], (0.0,1.0), (gamma = 10.0,))

prob = DIProblem(ode, cons)

alg = (ode = Euler(), lincompl = OneStepPJ())
sol = solve(prob, alg, dt = 0.001)





################
## Error plot ##
################

using CairoMakie 
exact_sol = [0.0, 1.0]

begin 
    f = Figure() 
    ax = Axis(f[1,1], title = "Trajectories")

    lines!(sol.t, sol[1,:], label = "u[1]")
    lines!(sol.t, sol[2,:], label = "u[2]")
    axislegend(ax)

    Axis(f[2,1], title = "Constraints", ylabel = "constraint (should be >= 0)", xlabel = "time")
    for con in cons
        lines!(sol.t, con.(sol.u))
    end
    dts = [2.0^(-i) for i in 1:16]
    err = [norm(solve(prob, alg, dt = dt)[end] - exact_sol) for dt in dts]


    ax = Axis(f[1:2,2], xscale = log2, yscale = log2)
    ax.xlabel = L"h"
    ax.ylabel = L"\Vert x^{\text{exact}}(T) - x(T) \Vert^2 "
    ax.title = "Error plot"
    lines!(dts, err)
    scatter!(dts, err)
        
    f
end
