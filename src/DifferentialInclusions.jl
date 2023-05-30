module DifferentialInclusions

    using SciMLBase, ForwardDiff, LinearAlgebra 

    include("DIProblem.jl")
    include("DIAlgorithm.jl")
    include("solvers/OSPJ.jl")

    export DIProblem

    export OSPJ, ProjectiveMethod

end
