
function add_grad!(u, f, dc)
    (idx, xi, c, dc) = dc
    for (i_src, i_des) in enumerate(idx)
        u[i_des] += f * dc[i_src]
    end
end


# find λ such that for z = Aλ + b such that 0 ≤ z ⟂ λ ≥ 0
struct LCProblem{At, Bt}
    A::At 
    b::Bt
end

Base.@kwdef struct PSOR{oT}
    ω::Float64 = 1.0
    reltol::Float64 = 1e-3
    abstol::Float64 = 1e-6
    maxiter::Int64 = 10_000
    obs::oT = nothing
end

PSOR(kwargs...) = PSOR(ω = 1.0; kwargs...)


struct PSORCache{vT} 
    λ::vT 
    λprev::vT
    z::vT
end 

function init_cache(::Type{LCProblem}, m, ::PSOR)
    return PSORCache(zeros(m), zeros(m), zeros(m))
end

function SciMLBase.solve(prob::LCProblem, alg::PSOR, cache = init_cache(LCProblem, length(prob.b), alg), stats = nothing)
    (; λ, λprev, z) = cache  
    (; A, b) = prob 
    (; ω) = alg

    m = length(b)
    λ .= zero(eltype(λ))
    λprev .= λ

    z .= b  # z = Aλ + b

    iter = 0

    err = Inf64 
    while ( err > alg.abstol &&
        err > alg.reltol * norm(λprev) / m &&
        iter < alg.maxiter ) 

        λprev .= λ

        for i in eachindex(λ)
            λ[i] = max(0, λ[i] - ω/A[i,i] * z[i])

            for j in eachindex(b)
                z[j] += A[j,i] * (λ[i] - λprev[i])
            end
        end

        err = 0.0 
        for i in eachindex(λ)
            err += (λ[i] - λprev[i])^2
        end
        err = sqrt(err / m)

        iter += 1
    end


    # we assume we have the DE stats...
    if !isnothing(stats)
        stats.nf2 += iter
    end

    return λ
end

struct PSORIntegrator{cT,oT,lcT <: LCProblem,lccT}
    cons::cT
    alg::PSOR{oT}
    lc::lcT
    lc_cache::lccT
end

function (alg::PSOR)(di::DIProblem)
    m = length(di.cons)
    lc = LCProblem(zeros(m,m), zeros(m))
    cache = init_cache(LCProblem, m, alg)
    return PSORIntegrator(di.cons, alg, lc, cache)
end

function create_lc!(lc::LCProblem, cons, u)
    for (i, dc) in enumerate(gradients(cons, u, true))
        lc.b[i] = dc[3]  # c_i(u)
        Dci = copy(dc[4])  # Dc_i(u)
        for (j, dcj) in enumerate(gradients(cons, u, true))
            if i <= j
                lc.A[i,j] = dot(Dci, dcj[4])  # Dc_i(u) ⋅ Dc_j(u)
                lc.A[j,i] = lc.A[i,j]  # Dc_i(u) ⋅ Dc_j(u)
            end
        end
    end
    return lc
end

function (cs::PSORIntegrator)(integrator)
    (; cons, alg, lc, lc_cache) = cs

    u = integrator.u
    dt = integrator.dt

    create_lc!(lc, cons, u)

    iter_before = integrator.stats.nf2  # get how much iterations we had before
    
    λ = solve(lc, cs.alg, lc_cache, integrator.stats)

    for (i, dc) in enumerate(gradients(cons, u))
        add_grad!(u, λ[i], dc)
    end

    iter = integrator.stats.nf2 - iter_before  # compute amount of added iterations 
    update_obs!(alg.obs, integrator, iter)

    nothing
end


Base.@kwdef struct PNGS{tO}
    ω::Float64 = 1.0
    reltol::Float64 = 1e-3
    abstol::Float64 = 1e-6
    maxiter::Int64 = 10_000
    obs::tO = nothing
end

PNGS(kwargs...) = PNGS(ω = 1.0; kwargs...)

struct PNGSIntegrator{cT,uT,tO}
    cons::cT
    alg::PNGS{tO}
    uprev::uT
end

function (alg::PNGS)(di::DIProblem)
    return PNGSIntegrator(di.cons, alg, copy(di.ode.u0))
end


update_obs!(obs, int, iters) = nothing
update_obs!(obs::Vector{Int64}, int, iters) = push!(obs, iters)


function projected_newton_raphson!(u, idx, xi, c, dc, ω = 1.0)
    if c < 0
        Δλ = -c * ω / dot(dc, dc)
        add_grad!(u, Δλ, (idx, xi, c, dc))
    end
    return nothing
end

function (cs::PNGSIntegrator)(integrator)
    err = Inf64
    (; cons, alg, uprev) = cs
    (; ω) = alg

    u = integrator.u

    iter = 0
    
    while ( err > cs.alg.abstol &&
            err > cs.alg.reltol * norm(uprev) / length(uprev) &&
            iter < cs.alg.maxiter ) 

        uprev .= u

        gds = gradients(cons, u, true)
        map(con -> projected_newton_raphson!(u, con..., ω), gds)
        
        iter += 1

        err = 0.0 
        for i in eachindex(u)
            err += (u[i] - uprev[i])^2
        end
        err = sqrt(err / length(u))
    end
    
    integrator.stats.nf2 += iter

    update_obs!(alg.obs, integrator, iter)
    nothing
end


PBD(;kwargs...) = PNGS(; ω = 1.0, maxiter = 1, kwargs...)
PGS(;kwargs...) = PSOR(; ω = 1.0, kwargs...)