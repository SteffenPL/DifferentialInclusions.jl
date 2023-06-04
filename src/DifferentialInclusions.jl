module DifferentialInclusions

    using SciMLBase, ForwardDiff, LinearAlgebra, DiffResults, NLSolversBase, StaticArrays

    include("DIProblem.jl")
    include("DIAlgorithm.jl")

    include("sparse_constraints.jl")
    
    include("solvers/splitting_methods.jl")
    include("solvers/penalty_methods.jl")

    export DIProblem

    export OSPJ, PSOR, PGS, PenaltyMethod
    export ProjectiveMethod

end
