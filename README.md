# DifferentialInclusions

[![Build Status](https://github.com/SteffenPL/DifferentialInclusions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SteffenPL/DifferentialInclusions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/SteffenPL/DifferentialInclusions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SteffenPL/DifferentialInclusions.jl)

> This package is work-in-progress! 

## Current state:

Implemented solvers for linear complementarity problems:
- `PSOR`: Projected Successive Over-Relaxation with parameter ω ∈ (0, 2)
- PGS: Projected Gauss-Seidel is a special case `PSOR(ω = 1)`

## Examples

### Simple example

We define a differential inclusion as a combination of an `ODEProblem` and constraints.
There are two types of time-stepping methods:
- Projective methods (which solve linear or nonlinear complementarity problems after each non-constrained step)
- Penalty methods (which add extra terms to obtain a smooth ODE problem)

In principle, each time-stepping methods can be combined with each constraint solver, however,
adaptive methods might lead to suboptimal numerical results.

```julia
cons = (
    u -> u[1] - 0.2, 
    u -> 2.0 - u[2], 
    u -> abs(u[1] - u[2]) - 1.0
)

function ode!(du, u, p, t)
    du[1] = -p.gamma
    du[2] = -p.gamma
end

ode = ODEProblem( ode!, [0.0, 2.0], (0.0,1.0), (gamma = 1.0,))
prob = DIProblem(ode, cons)

alg = ProjectiveMethod(OSPJ(),Euler())
sol = solve(prob, alg, dt = 0.001)


alg = ProjectiveMethod(PSOR(ω = 0.2),Euler())
sol_ = solve(prob, alg, dt = 0.001, progress = true)

alg = PenaltyMethod(Heun(), alpha = 1.0, gamma = 1e5)
sol__ = solve(prob, alg)
```

### Sparse constraints 

Simulation of $N$ spheres in an overdamped medium with non-overlap condition:
```julia
N = 40
u0 = 2 * ( rand(2, N) .- 0.5 )

cs = let
    R = 0.2
    f = (u) -> sum(x -> x^2, u[:,1] - u[:,2]) - R^2
    
    con = TSOnceDifferentiable(f, MMatrix{2,2}(zeros(2,2)))  
    l_inds = LinearIndices(u0)

    pairs = ((i,j) for i in 1:N for j in 1:i-1)
    pairs_indices = ( SMatrix{2,2,Int64,4}[ [l_inds[:,i] l_inds[:,j]] for (i,j) in pairs  ] )

    SparseConstraints(pairs_indices,  con)
end 


ode = ODEProblem((du,u,p,t) -> (@. du = -p.gamma * u), u0, (0.0,1.0), (gamma = 1.0,))
prob = DIProblem(ode, cs)
alg = ProjectiveMethod(PBD(), Euler())

sol_ = solve(ode, Euler(), dt = 1e-4);  #  3.369 ms (40044 allocations: 22.54 MiB)
sol = solve(prob, alg, dt = 1e-4);      # 85.476 ms (514091 allocations: 30.74 MiB)
```

Note that solving the `DIProblem` is, of course, slower than integrating just the `ODEProblem` because the ODE does not need to evaluate 
the constraints at each time step.
Thanks to the sparse constraints, the runtime is not `N * length(pairs) = 40 * 780` times slower, but instead only
around `30` times slower. The computational effort for evaluating the constraints is around `4 * length(pairs)`.
