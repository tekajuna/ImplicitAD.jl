module ImplicitAD

using ForwardDiff
using ReverseDiff
using ChainRulesCore
using LinearAlgebra: factorize, ldiv!, diag

# main function
export implicit, explicit_unsteady, implicit_unsteady, implicit_opt,  implicit_linear, apply_factorization, implicit_eigval, provide_rule

println("HITHERE! YOU ARE USING A LOCAL VERSION ON BRANCH  lagrange--it's awsome")
# ---------------------------------------------------------------------------
include("internals.jl")

export implicit
include("nonlinear.jl")

export explicit_unsteady, implicit_unsteady, explicit_unsteady_cache, implicit_unsteady_cache
include("unsteady.jl")

export implicit_linear, apply_factorization
include("linear.jl")

export implicit_eigval
include("eigenvalues.jl")

export provide_rule
include("external.jl")

export implicit_opt
include("lagrangian.jl")


end
