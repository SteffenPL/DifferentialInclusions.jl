using DifferentialInclusions, OrdinaryDiffEq
using Test

@testset "two stacked objects" begin
    

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

    alg = ProjectiveMethod(OSPJ(),Euler())
    sol = solve(prob, alg, dt = 0.0001)

    x_end = sol[end]
    @test isapprox(x_end[1], 0.0, atol = 1e-3)
    @test isapprox(x_end[2], 1.0, atol = 1e-3)

end


