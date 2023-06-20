
function projected_newton_raphson!(u, idx, xi, c, dc, ω = 1.0)
    if c < 0
        Δλ = -c * ω / dot(dc, dc)
        for (i_src, i_des) in enumerate(idx)
            u[i_des] += Δλ * dc[i_src]
        end
    end
    return nothing
end




Base.@kwdef struct PSOR{tO}
    ω::Float64 = 1.0
    reltol::Float64 = 1e-3
    abstol::Float64 = 1e-6
    maxiter::Int64 = 10_000
    obs::tO = nothing
end

PSOR(kwargs...) = PSOR(ω = 1.0; kwargs...)

struct PSORIntegrator{cT,uT,tO}
    cons::cT
    alg::PSOR{tO}
    uprev::uT
    u0::uT
end

function (alg::PSOR)(di::DIProblem)
    return PSORIntegrator(di.cons, alg, copy(di.ode.u0), copy(di.ode.u0))
end


function (cs::PSORIntegrator)(integrator)
    err = Inf64
    (; cons, alg, uprev, u0) = cs
    (; ω) = alg

    u = integrator.u

    u0 = copy(u)
    gds = gradients(cons, u0)

    iter = 0
    while ( err > cs.alg.abstol &&
            err > cs.alg.reltol * norm(uprev) / length(uprev) &&
            iter < cs.alg.maxiter ) 

        uprev .= u
        map(con -> projected_newton_raphson!(u, con..., ω), gds)

        iter += 1

        err = 0.0 
        for i in eachindex(u)
            err += (u[i] - uprev[i])^2
        end
        err = sqrt(err) / length(u)
    end
    
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

        gds = gradients(cons, u)
        map(con -> projected_newton_raphson!(u, con..., ω), gds)

        iter += 1

        err = 0.0 
        for i in eachindex(u)
            err += (u[i] - uprev[i])^2
        end
        err = sqrt(err) / length(u)
    end
    
    update_obs!(alg.obs, integrator, iter)
    nothing
end


PBD(;kwargs...) = PNGS(; ω = 1.0, maxiter = 1, kwargs...)
PGS(;kwargs...) = PSOR(; ω = 1.0, kwargs...)