Base.@kwdef struct PenaltyMethod <: AbstractDIAlgorithm
    alpha::Float64 = 2.0
    gamma::Float64 = 0.1
end

struct PenaltyMethodIntegrator{cT,cgT,aT}
end

function modify_ode(ode, cons, alg::PenaltyMethod)

    # TODO this is ugly, but it works for now
    m = length(cons)
    cons_grads = [DiffResults.GradientResult(ode.u0) for i in 1:m]

    # TODO: make general, ensure capture is done right, assume in-place for now! 
    function ode_mod(du, u, p, t)
        ode.f(du, u, p, t)  
        for i in eachindex(cons)
            cons_grads[i] = ForwardDiff.gradient!(cons_grads[i], cons[i], u)
            c = DiffResults.value(cons_grads[i])
            dc = DiffResults.gradient(cons_grads[i])
            apply!(alg, du, u, p, t, c, dc)
        end
        return nothing
    end
    ode_mod = remake(ode, f = ode_mod)
    return ode_mod
end

function apply!(alg::PenaltyMethod, du, u, p, t, c, dc)
    if c < 0 
        @. du -= alg.gamma * c ^ (alg.alpha-1) * dc
    end
    nothing
end

function (alg::PenaltyMethod)(di::DIProblem)
    return PenaltyMethodIntegrator()
end

