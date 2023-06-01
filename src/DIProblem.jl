struct DIProblem{odeT,consT}
    ode::odeT 
    cons::consT 
end

function DIProblem(ode, cons, alg) 
    ode_mod = modify_ode(ode, cons, alg)
    return DIProblem(ode_mod, cons)
end