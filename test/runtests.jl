using DifferentialInclusions
using Test

@testset "DifferentialInclusions.jl" begin
    # Write your tests here.
end




lin_comp_prob = LinearComplProblem(A, b)

lincompsolve = init!(lin_comp_prob)
lincompsolve.b = 10 
sol1 = solve!(lincompsolve)




du = ConstraintArray 



abstract type AbstractConstraint end 


function project!(constr, x, alg::NewtonRaphson)
    gradient!(∇x, constr, x) 
    Δx = s
end


using ForwardDiff
using OrdinaryDiffEq
using StaticArrays
using LinearAlgebra

struct DIProblem{odeT,consT}
    ode::odeT 
    cons::consT 
end

struct SparseConstraint{aT,fT,iT}
    approx::aT  
    cons::fT 
    idx::iT 
end
SparseConstraint(fnc, idx) = SparseConstraint(fnc, fnc, idx)

function (con::SparseConstraint)(x)
    u_ = subarray_tuple(x, con.idx)
    return con.cons(u_)
end


lower_bound(lb) = x -> x[1] - lb  
upper_bound(ub) = x -> ub - x[1]  

min_distance(d) = x -> abs(x[1] - x[2]) - d

cons = (
    SparseConstraint( lower_bound(0.0), (1,) ), 
    SparseConstraint( upper_bound(2.0), (2,) ), 
    SparseConstraint( min_distance(1.0), (1,2))
)

function ode!(du, u, p, t)
    du[1] = -p.gamma * u[1]
    du[2] = -p.gamma * u[2]
end

ode = ODEProblem( ode!, [0.0, 1.0], (0.0,1.0), (gamma = 10.0,))

prob = DIProblem(ode, cons)

using CairoMakie 

alg = (ode = Euler(), lincompl = PBDSolver(prob))
sol = solve(prob, alg, dt = 0.001)


exact_sol = [0.0, 1.0]

begin 
    f = Figure() 
    Axis(f[1,1])

    lines!(sol.t, sol[1,:])
    lines!(sol.t, sol[2,:])

    Axis(f[1,2])
    for con in cons
        lines!(sol.t, con.(sol.u))
    end
    f
end

dts = [2.0^(-i) for i in 1:14]
err = [norm(solve(prob, alg, dt = dt)[end] - exact_sol) for dt in dts]

begin 
    f = Figure()
    Axis(f[1,1], yscale = log2, xscale = log2, xlabel = L"h", ylabel = L"\Vert x^{\text{exact}}(T) - x(T) \Vert^2 ")
    lines!(dts, err)
    scatter!(dts, err)
    
    current_figure()
end 


function erro_est(dts, err)
    B = [log.(dts) ones(length(dts))]
    x = B \ log.(err)
    return x[1]
end
erro_est(dts, err)




sol = solve(prob, Euler(), dt = 0.1)

pbd = ProjectiveMethod(Euler(), IteratedProjections(NewtonRaphson()), dt = 0.1)


alg = (ode = Euler(), comp = IteratedProjections(NewtonRaphson()))





subarray_tuple(u, idx) = SVector{length(idx), Float64}(map( i -> u[i], idx))

function newton_raphson!(u, con::SparseConstraint) 
    u_ = subarray_tuple(u, con.idx)
    c = con.cons(u_) 
    if c < 0 
        u_ = subarray_tuple(u, con.idx)
        dc = ForwardDiff.gradient(con.cons, u_)
        Δu_ = -c / dot(dc, dc) * dc
        u_ = u_ + Δu_
        for (k,i) in enumerate(con.idx) 
            u[i] = u_[k]
        end
    end
    return nothing 
end 





struct PBDSolver{diT}
    di::diT 
end

function (cs::PBDSolver)(integrator)
    u = integrator.u 

    foreach( con -> newton_raphson!(u, con), cs.di.cons)
    nothing 
    #cs.di.g(u, cs.di.p, t)
end


function SciMLBase.solve(di::DIProblem, alg; kwargs...)

    each_step(u, t, integrator) = true 
    
    affect! = alg.lincompl

    cb = DiscreteCallback(each_step, affect!; save_positions = (false, true))

    return solve(di.ode, alg.ode; save_everystep = false, callback = cb, kwargs...)
end





int = init(ode, Euler(), dt = 0.1)

step!(int)

func(t, u, integrator)

FunctionCallingCallback(  )



abstract type AbstractDIAlgorithm end 

struct ProjectiveMethod{pA, oA} <: AbstractDIAlgorithm 
    proj_alg::pA 
    ode_alg::oA 
end

struct AugmentedLagrangian{pA, oA} <: AbstractDIAlgorithm 
    aug_alg::pA 
    ode_alg::oA 
end


abstract type AbstractDIIntegrator end

struct DIIntegrator{pI, oI} <: AbstractDIIntegrator 
    compl_int::pI 
    ode_int::oI 
end

DIProblem( (u,p,t) -> -u, (u,p,t) -> u, 0.0, (0.0,1.0), 0.1, 0.1 )

function step!(int::DIIntegrator) 
    step!(int.ode_int) 
    project!(int.compl_int)
end