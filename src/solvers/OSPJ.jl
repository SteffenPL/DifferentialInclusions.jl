
struct OSPJ end

struct OSPJIntegrator{cT}
    cons::cT
end

function (::OSPJ)(di::DIProblem) 
    return OSPJIntegrator(di.cons)
end 

function projected_newton_raphson!(u, con) 
    c = con(u) 
    if c < 0 
        dc = ForwardDiff.gradient(con, u)
        Δu = -c / dot(dc, dc) * dc
        u .+= Δu
    end
    return nothing 
end 

function (cs::OSPJIntegrator)(integrator)
    foreach(con -> projected_newton_raphson!(integrator.u, con), cs.cons)
end
