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

    alg = ProjectiveMethod(OSPJ(),Euler())
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

    cons = (
        u -> u[1] + u[2] - 0.0, 
        u -> u[2] + 1.0, 
    )

    function ode!(du, u, p, t)
        du[1] = 0.0
        du[2] = -p.gamma
    end

    # the speed of the object is first -p.gamma 
    # and after the contact with the slop, it will have speed -p.gamma / sqrt(2)
    # such that the speed in each axis direction is -p.gamma / 2

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


