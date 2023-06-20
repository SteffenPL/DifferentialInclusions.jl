using DifferentialInclusions, OrdinaryDiffEq
using Test

@testset "two stacked objects" begin
    
    using DifferentialInclusions, OrdinaryDiffEq

    cons = (
        u -> u[1] - 0.2, 
        u -> 2.0 - u[2], 
        u -> abs(u[1] - u[2]) - 1.0
    )

    function ode!(du, u, p, t)
        du[1] = -p.gamma * u[1]
        du[2] = -p.gamma * u[2]
    end

    ode = ODEProblem(ode!, [0.0, 2.0], (0.0,1.0), (gamma = 10.0,))
    prob = DIProblem(ode, cons)

    alg = ProjectiveMethod(PBD(),Euler())
    sol = solve(prob, alg, dt = 1e-5)


    x_end = sol[end]
    @test isapprox(x_end[1], 0.2, atol = 1e-3)
    @test isapprox(x_end[2], 1.2, atol = 1e-3)


    alg = ProjectiveMethod(PGS(),Euler())
    sol = solve(prob, alg, dt = 1e-5)

    x_end = sol[end]
    @test isapprox(x_end[1], 0.2, atol = 1e-3)
    @test isapprox(x_end[2], 1.2, atol = 1e-3)

end

@testset "Overdamped fall onto slope" begin 

    using DifferentialInclusions, OrdinaryDiffEq

    cons = (
        u -> u[1] + u[2] - 0.0, 
        u -> u[2] + 1.0, 
    )

    function ode!(du, u, p, t)
        du[1] = 0.0
        du[2] = -p.gamma
    end

    ode = ODEProblem( ode!, [0.0, 1.0], (0.0,2.0), (gamma = 1.0,))
    prob = DIProblem(ode, cons)

    alg = ProjectiveMethod(PGS(),Euler())
    sol = solve(prob, alg, dt = 1e-5)

    x_end = sol[end]
    @test isapprox(x_end[1], ode.p.gamma / 2, atol = 1e-3)
    @test isapprox(x_end[2], -ode.p.gamma / 2, atol = 1e-3)
end

@testset "penalty method" begin
    
    cons = (
        u -> u[1] - 0.2, 
        u -> 2.0 - u[2], 
        u -> abs(u[1] - u[2]) - 1.0
    )

    function ode!(du, u, p, t)
        du[1] = -p.gamma * u[1]
        du[2] = -p.gamma * u[2]
    end

    ode = ODEProblem(ode!, [0.0, 2.0], (0.0,1.0), (gamma = 10.0,))
    
    alg = PenaltyMethod(Heun(), alpha = 2.0, gamma = 1e5)
    prob = DIProblem(ode, cons)
    sol = solve(prob, alg)


    x_end = sol[end]
    @test isapprox(x_end[1], 0.2, atol = 1e-3)
    @test isapprox(x_end[2], 1.2, atol = 1e-3)
end


using DifferentialInclusions: TSOnceDifferentiable, SparseConstraints
using StaticArrays

@testset "10 spheres attracted to center" begin

        
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

    
    
    # test if overlap is not too much (probablistic test, might fail!)
    @test minimum( [ c for (idx,xi,c,dx) in DifferentialInclusions.gradients(cs, sol[end])] ) > -1e-3
end