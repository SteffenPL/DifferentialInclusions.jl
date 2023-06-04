struct SparseConstraints{iT, cT}  
    indices::iT 
    cons::cT
end

function SparseConstraints(indices, cons::Function, x; autodiff = :forward)
    df = OnceDifferentiable(cons, x; autodiff)
    return SparseConstraints(indices, df)
end

@inline subarray(x, idx::Tuple) = SVector( (x[i] for i in idx)... )
@inline subarray(x, idx::StaticArray) = SMatrix{size(idx)...}( (x[i] for i in idx)... )

function gradients(sp, x, idx)
    xi = subarray(x, idx)
    c = value!(sp.cons, xi)
    if c < 0 
        gradient!(sp.cons, xi)
    end
    return (idx, xi, value(sp.cons), gradient(sp.cons))
end

function gradients(sp::SparseConstraints, x)
    return (gradients(sp, x, idx) for idx in sp.indices)
end

function gradients(sp::Vector, x)
    return ( (eachindex(x), x, value!(c, x), gradient!(c, x)) for c in sp )
end

# this is not in-place, added for simplicity
function _gradient!(dr, f, x)
    dr = ForwardDiff.gradient!(dr, f, x)
    return (DiffResults.value(dr), DiffResults.gradient(dr))
end

function gradients(sp, x)
    dr = DiffResults.GradientResult(x)
    return ((eachindex(x), x, _gradient!(dr, c, x)...) for c in sp)
end




#####  This type should not be needed in future, probably exists somewhere anyway!
# Type-stable once differentiable  
mutable struct TSOnceDifferentiable{Tf, TF, TX} <: AbstractObjective
    f::Tf # objective function 
    diffres::TF # diff result cache
    x_f::TX # x used to evaluate f
    x_df::TX # x used to evaluate df
end

using DiffResults, ForwardDiff

function TSOnceDifferentiable(f, x)
    x_nan = x .* NaN 
    return TSOnceDifferentiable(f, DiffResults.GradientResult(x), x_nan, copy(x_nan))
end

function NLSolversBase.value!(ts::TSOnceDifferentiable, x)
    if ts.x_f != x
        ts.diffres = DiffResults.value!(ts.diffres, ts.f(x))
        ts.x_f .= x
    end
    return DiffResults.value(ts.diffres)
end

function NLSolversBase.gradient!(ts::TSOnceDifferentiable, x)
    if ts.x_df != x
        ts.diffres = ForwardDiff.gradient!(ts.diffres, ts.f, x)
        ts.x_df .= x
        ts.x_f .= x
    end

    return DiffResults.gradient(ts.diffres)
end

NLSolversBase.value(ts::TSOnceDifferentiable) = DiffResults.value(ts.diffres)
NLSolversBase.gradient(ts::TSOnceDifferentiable) = DiffResults.gradient(ts.diffres)