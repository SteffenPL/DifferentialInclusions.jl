struct DIProblem{odeT,consT}
    ode::odeT 
    cons::consT 
end

modify_ode(ode, cons, alg) = ode