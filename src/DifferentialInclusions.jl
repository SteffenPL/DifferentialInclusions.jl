module DifferentialInclusions

    using SciMLBase, ForwardDiff, LinearAlgebra 

    include("DIProblem.jl")
    include("DIAlgorithm.jl")
    include("solvers/splitting_methods.jl")

    export DIProblem

    export OSPJ, PSOR, PGS
    export ProjectiveMethod

end
