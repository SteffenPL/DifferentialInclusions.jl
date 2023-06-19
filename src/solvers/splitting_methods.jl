
Base.@kwdef struct OSPJ
    fwd::Bool = true
end

struct OSPJIntegrator{cT}
    cons::cT
    alg::OSPJ
end

function (alg::OSPJ)(di::DIProblem)
    return OSPJIntegrator(di.cons, alg)
end

function projected_newton_raphson!(u, idx, xi, c, dc)
    if c < 0
        Δu = -c / dot(dc, dc) * dc
        for (i_src, i_des) in enumerate(idx)
            u[i_des] += Δu[i_src]
        end
    end
    return nothing
end

function (cs::OSPJIntegrator)(integrator)
    u = integrator.u
    if cs.alg.fwd
        gds = gradients(cs.cons, u)
        map(con -> projected_newton_raphson!(u, con...), gds)
        #map(con -> projected_newton_raphson!(u, con...), gds)
    else
        gds_r = Iterators.reverse(gradients(cs.cons, u))
        map(con -> projected_newton_raphson!(u, con...), gds_r)
    end
end


Base.@kwdef struct PSOR
    ω::Float64 = 1.0
    reltol::Float64 = 1e-3
    abstol::Float64 = 1e-6
    maxiter::Int64 = 10_000
end

PGS(kwargs...) = PSOR(ω = 1.0; kwargs...)

struct PSORIntegrator{cT,cgT}
    cons::cT
    alg::PSOR
    cons_grad::cgT
    λ::Vector{Float64}
    λ_prev::Vector{Float64}
end

function (alg::PSOR)(di::DIProblem)
    m = length(di.cons)

    cons_grad = [DiffResults.GradientResult(di.ode.u0) for i in 1:m]
    return PSORIntegrator(di.cons, alg, cons_grad, zeros(m), zeros(m))
end

function (cs::PSORIntegrator)(integrator)
    err = Inf64
    (; cons, cons_grad, alg, λ, λ_prev) = cs
    (; ω) = alg
    m = length(cons)

    u = integrator.u
    du = get_du(integrator)
    dt = integrator.dt

    for (i, con) in enumerate(cons)
        cons_grad[i] = ForwardDiff.gradient!(cons_grad[i], con, u)
    end

    λ .= zero(eltype(λ))
    iter = 0

    c(i) = DiffResults.value(cons_grad[i])
    dc(i) = DiffResults.gradient(cons_grad[i])

    q(i) = c(i) + dt * dot(dc(i), du)
    M(i,j) = dot(dc(i), dc(j))


    while ( err > cs.alg.abstol &&
            err > cs.alg.reltol * norm(λ) &&
            iter < cs.alg.maxiter ) 

        λ_prev .= λ
        for i in 1:m
            λ[i] = max(0, λ[i] - ω / M(i,i) * (q(i) + sum(M(j,i) * λ[j] for j in 1:m)) )
        end

        iter += 1
        err = sqrt(sum(x -> x^2, λ - λ_prev))
    end
    
    for i in 1:m
        u[:] .+= λ[i] * dc(i)  # the gradient might not have same size as u! 
    end
    nothing
end
