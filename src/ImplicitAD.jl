module ImplicitAD

using ForwardDiff
using ReverseDiff
using ChainRulesCore
using LinearAlgebra: factorize, ldiv!, diag

# main function
export implicit, explicit_unsteady, implicit_unsteady, implicit_opt,  implicit_linear, apply_factorization, implicit_eigval, provide_rule

println("HITHERE! YOU're USING A LOCAL VERSION ON BRANCH  lagrange")
# ---------------------------------------------------------------------------

# ------- Unpack/Pack ForwardDiff Dual ------

fd_value(x) = ForwardDiff.value.(x)
fd_partials(x) = reduce(vcat, transpose.(ForwardDiff.partials.(x)))

"""
unpack ForwardDiff Dual return value and derivative.
"""
function unpack_dual(x)
    xv = fd_value(x)
    dx = fd_partials(x)
    return xv, dx
end

"""
Create a ForwardDiff Dual with value yv, derivatives dy, and Dual type T
"""
pack_dual(yv::AbstractFloat, dy, T) = ForwardDiff.Dual{T}(yv, ForwardDiff.Partials(Tuple(dy)))
pack_dual(yv::AbstractVector, dy, T) = ForwardDiff.Dual{T}.(yv, ForwardDiff.Partials.(Tuple.(eachrow(dy))))

# -----------------------------------------

# ---------- core methods -----------------

"""
Compute Jacobian vector product b = -B*xdot where B_ij = ∂r_i/∂x_j
This takes in the dual directly for x = (xv, xdot) since it is already formed that way.
"""
function jvp(residual, y, xd, p)

    # evaluate residual function
    rd = residual(y, xd, p)  # constant y

    # extract partials
    b = -fd_partials(rd)

    return b

end

# explicit unsteady
function jvp(perform_step, yprevd, t, tprev, xd, p)

    # evaluate residual function
    yd = perform_step(yprevd, t, tprev, xd, p)  # constant y

    # extract partials
    return fd_partials(yd)
end

# implicit unsteady
function jvp(residual, y, yprevd, t, tprev, xd, p)

    # evaluate residual function
    rd = residual(y, yprevd, t, tprev, xd, p)  # constant y

    # extract partials
    b = -fd_partials(rd)

    return b
end


"""
Compute vector Jacobian product c = B^T v = (v^T B)^T where B_ij = ∂r_i/∂x_j and return c
"""
function vjp(residual, y, x, p, v)
    return ReverseDiff.gradient(xtilde -> v' * residual(y, xtilde, p), x)  # instead of using underlying functions in ReverseDiff just differentiate -v^T * r with respect to x to get -v^T * dr/dx
end

# explicit unsteady
function vjp(perform_step, yprev, t, tprev, x, p, v)
    return ReverseDiff.gradient((yprevtilde, xtilde) -> v' * perform_step(yprevtilde, t, tprev, xtilde, p), (yprev, x))
end

# implicit unsteady
function vjp(residual, y, yprev, t, tprev, x, p, v)
    return ReverseDiff.gradient((yprevtilde, xtilde) -> v' * residual(y, yprevtilde, t, tprev, xtilde, p), (yprev, x))
end


"""
compute A_ij = ∂r_i/∂y_j
"""
function drdy_forward(residual, y::AbstractVector, x, p)
    A = ForwardDiff.jacobian(ytilde -> residual(ytilde, x, p), y)
    return A
end

function drdy_forward(residual, y::Number, x, p)  # 1D case
    A = ForwardDiff.derivative(ytilde -> residual(ytilde, x, p), y)
    return A
end

# implicit unsteady
function drdy_forward(residual, y::AbstractVector, yprev, t, tprev, x, p)
    A = ForwardDiff.jacobian(ytilde -> residual(ytilde, yprev, t, tprev, x, p), y)
    return A
end

function drdy_forward(residual, y::Number, yprev, t, tprev, x, p)  # 1D case
    A = ForwardDiff.derivative(ytilde -> residual(ytilde, yprev, t, tprev, x, p), y)
    return A
end


"""
Linear solve A x = b  (where A is computed in drdy and b is computed in jvp).
"""
linear_solve(A, b) = A\b

linear_solve(A::Number, b) = b / A  # scalar division for 1D case

# -----------------------------------------



# ---------- Overloads for implicit ---------

"""
    implicit(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).

# Arguments
- `solve::function`: y = solve(x, p). Solve implicit function returning state variable y, for input variables x, and fixed parameters p.
- `residual::function`: Either r = residual(y, x, p) or in-place residual!(r, y, x, p). Set residual r (scalar or vector), given state y (scalar or vector), variables x (vector) and fixed parameters p (tuple).
- `x::vector{float}`: evaluation point.
- `p::tuple`: fixed parameters. default is empty tuple.
- `drdy::function`: drdy(residual, y, x, p).  Provide (or compute yourself): ∂r_i/∂y_j.  Default is forward mode AD.
- `lsolve::function`: lsolve(A, b).  Linear solve A x = b  (where A is computed in drdy and b is computed in jvp, or it solves A^T x = c where c is computed in vjp).  Default is backslash operator.
"""
function implicit(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

    # ---- check for in-place version and wrap as needed -------
    new_residual = residual
    if applicable(residual, 1.0, 1.0, 1.0, 1.0)  # in-place

        function residual_wrap(yw, xw, pw)  # wrap residual function in a explicit form for convenience and ensure type of r is appropriate
            T = promote_type(eltype(xw), eltype(yw))
            rw = zeros(T, length(yw))  # match type of input variables
            residual(rw, yw, xw, pw)
            return rw
        end
        new_residual = residual_wrap
    end

    return implicit(solve, new_residual, x, p, drdy, lsolve)
end



# If no AD, just solve normally.
implicit(solve, residual, x, p, drdy, lsolve) = solve(x, p)



# Overloaded for ForwardDiff inputs, providing exact derivatives using
# Jacobian vector product.
function implicit(solve, residual, x::AbstractVector{<:ForwardDiff.Dual{T}}, p, drdy, lsolve) where {T}

    # evaluate solver
    xv = fd_value(x)
    yv = solve(xv, p)

    # solve for Jacobian-vector product
    b = jvp(residual, yv, x, p)

    # compute partial derivatives
    A = drdy(residual, yv, xv, p)

    # linear solve
    ydot = lsolve(A, b)

    # repack in ForwardDiff Dual
    return pack_dual(yv, ydot, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(implicit), solve, residual, x, p, drdy, lsolve)

    # evaluate solver (create local copy of the output to guard against `y` getting overwritten)
    y = copy(solve(x, p))

    function pullback(ybar)
        A = drdy(residual, y, x, p)
        u = lsolve(A', ybar)
        xbar = vjp(residual, y, x, p, -u)
        return NoTangent(), NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent()
    end

    return y, pullback
end


# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules implicit(solve, residual, x::TrackedArray, p, drdy, lsolve)
ReverseDiff.@grad_from_chainrules implicit(solve, residual, x::AbstractVector{<:TrackedReal}, p, drdy, lsolve)

# ------ Overloads for explicit_unsteady ----------

"""
    explicit_unsteady(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

Make an explicit time-marching analysis AD compatible (specifically with ForwardDiff and
ReverseDiff).

# Arguments:
 - `solve::function`: function `y = solve(x, p)`.  Perform a time marching analysis,
    and return the matrix `[y[1] y[2] y[3] ... y[N]; t[1], t[2], t[3] ... t[N]]` where `y[i]`
    is the state vector at time step `i` and `t[i]` is the corresponding time, for input
    variables `x`, initial conditions `y0`, and fixed variables `p`.
 - `perform_step::function`: Either `y[i] = perform_step(y[i-1], t[i], t[i-1], x, p)` or
    in-place `perform_step!(y[i], y[i-1], t[i], t[i-1], x, p)`. Set the next set of state
    variables `y[i]` (scalar or vector), given the previous state `y[i-1]` (scalar or vector),
    current time `t[i]`, previous time `t[i-1]`, variables `x` (vector), and fixed parameters `p` (tuple).
 - `x`: evaluation point
 - `p`: fixed parameters, default is empty tuple.
"""
function explicit_unsteady(solve, perform_step, x, p=())

    # ---- check for in-place version and wrap as needed -------
    new_perform_step = perform_step
    if applicable(perform_step, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # in-place

        function explicit_perform_step_wrap(yprevw, tw, tprevw, xw, pw)  # wrap perform_step function in a explicit form for convenience and ensure type of r is appropriate
            T = promote_type(eltype(yprevw), typeof(tw), typeof(tprevw), eltype(xw))
            yw = zeros(T, length(yprevw))  # match type of input variables
            perform_step(yw, yprevw, tw, tprevw, xw, pw)
            return yw
        end
        new_perform_step = explicit_perform_step_wrap
    end

    return _explicit_unsteady(solve, new_perform_step, x, p)
end

# If no AD, just solve normally.
_explicit_unsteady(solve, perform_step, x, p) = solve(x, p)

# Overloaded for ForwardDiff inputs, providing exact derivatives using Jacobian vector product.
function _explicit_unsteady(solve, perform_step, x::AbstractVector{<:ForwardDiff.Dual{T}}, p) where {T}

    # evaluate solver
    xv = fd_value(x)
    ytv = solve(xv, p)

    # get solution dimensions
    ny = size(ytv, 1) - 1
    nt = size(ytv, 2)

    # initialize output
    yt = similar(ytv, ForwardDiff.Dual{T}, ny+1, nt)

    # --- Initial Time Step --- #

    # solve for Jacobian-vector products
    @views ydot = jvp(perform_step, ytv[1:ny,1], ytv[ny+1,1], ytv[ny+1,1], x, p)

    # # repack in ForwardDiff Dual
    @views yt[1:ny,1] = pack_dual(ytv[1:ny,1], ydot, T)
    yt[ny+1,1] = ytv[ny+1,1]

    # --- Additional Time Steps --- #

    for i = 2:nt

        # solve for Jacobian-vector product
        @views ydot = jvp(perform_step, yt[1:ny,i-1], ytv[ny+1,i], ytv[ny+1,i-1], x, p)

        # repack in ForwardDiff Dual
        @views yt[1:ny,i] = pack_dual(ytv[1:ny,i], ydot, T)
        yt[ny+1,i] = ytv[ny+1,i]

    end

    return yt
end

# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(_explicit_unsteady), solve, perform_step, x, p)

    # evaluate solver
    yt = solve(x, p)

    # get solution dimensions
    ny = size(yt, 1) - 1
    nt = size(yt, 2)

    # create local copy of the output to guard against `yt` getting overwritten
    yt = copy(yt)

    function pullback(ytbar)

        if nt > 1

            # --- Final Time Step --- #
            @views λ = ytbar[1:ny,nt]
            @views Δybar, Δxbar = vjp(perform_step, yt[1:ny,nt-1], yt[ny+1,nt], yt[ny+1,nt-1], x, p, λ)
            xbar = Δxbar
            @views ytbar[1:ny,nt-1] += Δybar

            # --- Additional Time Steps --- #
            for i = nt-1:-1:2
                @views λ = ytbar[1:ny,i]
                @views Δybar, Δxbar = vjp(perform_step, yt[1:ny,i-1], yt[ny+1,i], yt[ny+1,i-1], x, p, λ)
                xbar += Δxbar
                @views ytbar[1:ny,i-1] += Δybar
            end

            # --- Initial Time Step --- #
            @views λ = ytbar[1:ny,1]
            @views Δybar, Δxbar = vjp(perform_step, yt[1:ny,1], yt[ny+1,1], yt[ny+1,1], xbar, p, λ)
            xbar += Δxbar

        else

            # --- Initial Time Step --- #
            @views λ = ytbar[1:ny,1]
            @views Δybar, Δxbar = vjp(perform_step, yt[1:ny,1], yt[ny+1,1], yt[ny+1,1], xbar, p, λ)
            xbar = Δxbar

        end

        return NoTangent(), NoTangent(), NoTangent(), xbar, NoTangent()
    end

    return yt, pullback
end

# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules _explicit_unsteady(solve, perform_step, x::TrackedArray, p)
ReverseDiff.@grad_from_chainrules _explicit_unsteady(solve, perform_step, x::AbstractVector{<:TrackedReal}, p)


# ------ Overloads for implicit_unsteady ----------

"""
    implicit_unsteady(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

Make a implicit time-marching analysis AD compatible (specifically with ForwardDiff and
ReverseDiff).

# Arguments:
 - `solve::function`: function `y = solve(x, p)`.  Perform a time marching analysis,
    and return the matrix `[y[1] y[2] y[3] ... y[N]; t[1], t[2], t[3] ... t[N]]` where `y[i]`
    is the state vector at time step `i` and `t[i]` is the corresponding time, for input
    variables `x`, and fixed variables `p`.
 - `residual::function`: Either `r[i] = residual(t[i], t[i-1], y[i], y[i-1], x, p)` or
    in-place `residual!(r[i], t[i], t[i-1], y[i], y[i-1], x, p)`. Set current residual
    `r[i]` (scalar or vector), given current state `y[i]` (scalar or vector),
    previous state `y[i-1]` (scalar or vector), variables `x` (vector), and fixed
    parameters `p` (tuple).
 - `x`: evaluation point
 - `p`: fixed parameters, default is empty tuple.
 - `drdy`: drdy(residual, y, x, p). Provide (or compute yourself): ∂ri/∂yj.  Default is
    forward mode AD.
 - `lsolve::function`: `lsolve(A, b)`. Linear solve `A x = b` (where `A` is computed in
    `drdy` and `b` is computed in `jvp`, or it solves `A^T x = c` where `c` is computed
    in `vjp`). Default is backslash operator.
"""
function implicit_unsteady(solve, residual, x, p=(); drdy=drdy_forward, lsolve=linear_solve)

    # ---- check for in-place version and wrap as needed -------
    new_residual = residual
    if applicable(residual, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # in-place

        function unsteady_residual_wrap(yw, yprevw, tw, tprevw, xw, pw)  # wrap residual function in a explicit form for convenience and ensure type of r is appropriate
            T = promote_type(typeof(tw), typeof(tprevw), eltype(yw), eltype(yprevw), eltype(xw))
            rw = zeros(T, length(yw))  # match type of input variables
            residual(rw, yw, yprevw, tw, tprevw, xw, pw)
            return rw
        end
        new_residual = unsteady_residual_wrap
    end

    return implicit_unsteady(solve, new_residual, x, p, drdy, lsolve)
end

# If no AD, just solve normally.
implicit_unsteady(solve, residual, x, p, drdy, lsolve) = solve(x, p)

# Overloaded for ForwardDiff inputs, providing exact derivatives using Jacobian vector product.
function implicit_unsteady(solve, residual, x::AbstractVector{<:ForwardDiff.Dual{T}}, p, drdy, lsolve) where {T}

    # evaluate solver
    xv = fd_value(x)
    ytv = solve(xv, p)

    # get solution dimensions
    ny = size(ytv, 1) - 1
    nt = size(ytv, 2)

    # initialize output
    yt = similar(ytv, ForwardDiff.Dual{T}, ny+1, nt)

    # --- Initial Time Step --- #

    # solve for Jacobian-vector products
    @views b = jvp(residual, ytv[1:ny,1], ytv[1:ny,1], ytv[ny+1,1], ytv[ny+1,1], x, p)

    # compute partial derivatives
    @views A = drdy(residual, ytv[1:ny,1], ytv[1:ny,1], ytv[ny+1,1], ytv[ny+1,1], xv, p)

    # linear solve
    ydot = lsolve(A, b)

    # # repack in ForwardDiff Dual
    @views yt[1:ny,1] = pack_dual(ytv[1:ny,1], ydot, T)
    yt[ny+1,1] = ytv[ny+1,1]

    # --- Additional Time Steps --- #

    for i = 2:nt

        # solve for Jacobian-vector product
        @views b = jvp(residual, ytv[1:ny,i], yt[1:ny,i-1], ytv[ny+1,i], ytv[ny+1,i-1], x, p)

        # compute partial derivatives
        @views A = drdy(residual, ytv[1:ny,i], ytv[1:ny,i-1], ytv[ny+1,i], ytv[ny+1,i-1], xv, p)

        # linear solve
        ydot = lsolve(A, b)

        # repack in ForwardDiff Dual
        @views yt[1:ny,i] = pack_dual(ytv[1:ny,i], ydot, T)
        yt[ny+1,i] = ytv[ny+1,i]

    end

    return yt
end

# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(implicit_unsteady), solve, residual, x, p, drdy, lsolve)

    # evaluate solver
    yt = solve(x, p)

    # get solution dimensions
    ny = size(yt, 1) - 1
    nt = size(yt, 2)

    # create local copy of the output to guard against `yt` getting overwritten
    yt = copy(yt)

    function pullback(ytbar)

        if nt > 1

            # --- Final Time Step --- #
            @views A = drdy(residual, yt[1:ny,nt], yt[1:ny,nt-1], yt[ny+1,nt], yt[ny+1,nt-1], x, p)
            @views λ = lsolve(A', ytbar[1:ny,nt])
            @views Δybar, Δxbar = vjp(residual, yt[1:ny,nt], yt[1:ny,nt-1], yt[ny+1,nt], yt[ny+1,nt-1], x, p, -λ)
            xbar = Δxbar
            @views ytbar[1:ny,nt-1] += Δybar

            # --- Additional Time Steps --- #
            for i = nt-1:-1:2
                @views A = drdy(residual, yt[1:ny,i], yt[1:ny,i-1], yt[ny+1,i], yt[ny+1,i-1], x, p)
                @views λ = lsolve(A', ytbar[1:ny,i])
                @views Δybar, Δxbar = vjp(residual, yt[1:ny,i], yt[1:ny,i-1], yt[ny+1,i], yt[ny+1,i-1], x, p, -λ)
                xbar += Δxbar
                @views ytbar[1:ny,i-1] += Δybar
            end

            # --- Initial Time Step --- #
            @views A = drdy(residual, yt[1:ny,1], yt[1:ny,1], yt[ny+1,1], yt[ny+1,1], x, p)
            @views λ = lsolve(A', ytbar[1:ny,1])
            @views Δybar, Δxbar = vjp(residual, yt[1:ny,1], yt[1:ny,1], yt[ny+1,1], yt[ny+1,1], xbar, p, -λ)
            xbar += Δxbar

        else

            # --- Initial Time Step --- #
            @views A = drdy(residual, yt[1:ny,1], yt[1:ny,1], yt[ny+1,1], yt[ny+1,1], x, p)
            @views λ = lsolve(A', ytbar[1:ny,1])
            @views Δybar, Δxbar = vjp(residual, yt[1:ny,1], yt[1:ny,1], yt[ny+1,1], yt[ny+1,1], xbar, p, -λ)
            xbar += Δxbar

        end

        return NoTangent(), NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent()
    end

    return yt, pullback
end

# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules implicit_unsteady(solve, residual, x::TrackedArray, p, drdy, lsolve)
ReverseDiff.@grad_from_chainrules implicit_unsteady(solve, residual, x::AbstractVector{<:TrackedReal}, p, drdy, lsolve)


# ---------- differentiable optimization problems----------------
implicit_opt(solve, lagrangian, P, opts,NDV;method="Hessian") = solve(P, opts)

"""
    implicit_opt(solve, objcon, P, opts,NDV)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).

# Arguments
- `solve::function`: xL = solve(P, opts). Solve optimization problem returning state variables x and L, for input variables P, and fixed optional parameters opts.
- `objcon::vector{function}`: 
- `P::vector{float}`: evaluation point; constant parameters given to the optimization problem
- `opts::tuple`: optional fixed parameters to solve. default is empty tuple.
- `NDV::int`: number of design variables, to distinguish output state from intermediate lagrange multipliers
"""

function implicit_opt(solve, objcon,P::AbstractVector{<:ForwardDiff.Dual{T}}, opts,NDV; method="Hessian") where {T}
    Pv = fd_value(P) # get parameter values, isolated from derivatives
    xLv = solve(Pv,opts)    # solve for design variables and multipliers, given input parameters
    if method == "Hessian" # This method allows for a simpler residual function, but is inefficient with many inactive constraints or input parameters
        lagrangian=construct_lagrangian(objcon,xLv,Pv,NDV)
        H = ForwardDiff.hessian(lagrangian,vcat(xLv,Pv))

        # Initally, all variables are relevant, and we have indices for each DV and LM
        relevant = BitSet(1:length(xLv))

        # Remove indices of any lagrange multiplier that is 
        for i=NDV+1:length(xLv)
            if xLv[i] <1e-5
                delete!(relevant,i)
            end
        end
        relevant = collect(relevant)
        xLvR = xLv[relevant]
        
        # Extract Residual-InputParameters derivatives from Hessian of Lagrangian
        if length(P)==1
            dRdP = H[relevant,end]
        else
            dRdP = H[relevant,[length(xLv)+1,end]] #TODO: Fix this
        end 
        
        # Extract Residual-DVs/multipliers derivatives from Hessian of Lagrangian
        dRdxL = H[relevant,relevant] # 

        ydot = -dRdxL\dRdP
    end
    if method == "residual"

        
        # Initally, all variables are relevant, and we have indices for each DV and LM
        relevant = BitSet(1:length(xLv))

        # Remove indices of any lagrange multiplier that is 
        for i=NDV+1:length(xLv)
            if xLv[i] <1e-5
                delete!(relevant,i)
            end
        end
        relevant = collect(relevant)
        
        # reconstruct output vector xLv to include only relevant pieces
        xLvR = xLv[relevant]
        residual=residual_constructor(relevant,objcon,xLv,Pv,NDV)
        
        # solve for Jacobian-vector product
        b = jvp(residual, xLvR, P, opts)
        
        # compute partial derivatives
        A = drdy_forward(residual, xLvR, Pv, opts)
        
        # linear solve
        ydot = linear_solve(A, b)

    end

    return pack_dual(xLvR[1:NDV], ydot[1:NDV,begin:end], T) # return relevant partials
end

function construct_lagrangian(objcon,xLv,Pv,NDV)
    M = length(xLv)
    N = length(Pv)
    
    function Lagrangian(X)
        Lagr = objcon[1](X[begin:NDV],X[M+1:end]) # functions are f(x,P)
        for i = 2:length(objcon) # for each L. Multiplier
            Lagr += objcon[i](X[begin:NDV],X[M+1:end])* X[NDV+i-1]
        end
        return Lagr
    end
    
    return Lagrangian
    
end
function residual_constructor(REL,objcon,xLv,Pv,NDV)
    M = length(xLv)
    N = length(Pv)
    
    function residual(xL,P,A)
        # Following ImplicitAD ideas, Y is going to be length of DV+non-zero lambdas
        # X is the number of parameters, and we should know in advance where these are in 
        # obj and cons 
        # obj is a function, and we could have a list of con functions, and then index into them. . .
        # Y = [x,y,lambda1,lambda2] design variables and lagrange multipliers
        # X = [P] some constant parameter
        # dL/dx
        # dL/dy
        # dL/dP
        G =[]
        
        push!(G,ForwardDiff.gradient(objcon[1],vcat(xL[begin:NDV],P)))
        for i = NDV+1:length(REL)
            push!(G,xL[REL[i]]*ForwardDiff.gradient(objcon[REL[i]-NDV+1],vcat(xL[begin:NDV],P)))
        end
        
        G = hcat(G...) # All the gradients 
        R = []#Array{Float64}(undef, NDV+length(REL))
        for i = 1:NDV
            push!(R,sum(G[i,:])) # Derivative of Lagr. wrt. ith DV
        end

        for i = NDV+1:length(REL)
            push!(R, objcon[REL[i]-NDV+1](vcat(xL[begin:NDV],P)))
        end

        R = hcat(R...)
        return R
        
    end 
    return residual
end


# ------ linear case ------------

"""
    apply_factorization(A, factfun)

Apply a matrix factorization to the primal portion of a dual number.
Avoids user from needing to add ForwardDiff as a dependency.
`Afactorization = factfun(A)`
"""
function apply_factorization(A::AbstractArray{<:ForwardDiff.Dual{T}}, factfun) where {T}
    Av = fd_value(A)
    return factfun(Av)
end

# just apply factorization normally for case with no Duals
apply_factorization(A, factfun) = factfun(A)

# default factorization
apply_factorization(A) = apply_factorization(A, factorize)



"""
    implicit_linear(A, b; lsolve=linear_solve, Af=factorize)

Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).
This version is for linear equations Ay = b

# Arguments
- `A::matrix`, `b::vector`: components of linear system ``A y = b``
- `lsolve::function`: lsolve(A, b). Function to solve the linear system, default is backslash operator.
- `Af::factorization`: An optional factorization of A, useful to override default factorize, or if multiple linear solves will be performed with same A matrix.
"""
implicit_linear(A, b; lsolve=linear_solve, Af=nothing) = implicit_linear(A, b, lsolve, Af)


# If no AD, just solve normally.
implicit_linear(A, b, lsolve, Af) = isnothing(Af) ? lsolve(A, b) : lsolve(Af, b)

# catch three cases where one or both contain duals
implicit_linear(A::AbstractArray{<:ForwardDiff.Dual{T}}, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, Af) where {T} = linear_dual(A, b, lsolve, Af, T)
implicit_linear(A, b::AbstractArray{<:ForwardDiff.Dual{T}}, lsolve, Af) where {T} = linear_dual(A, b, lsolve, Af, T)
implicit_linear(A::AbstractArray{<:ForwardDiff.Dual{T}}, b, lsolve, Af) where {T} = linear_dual(A, b, lsolve, Af, T)

# Both A and b contain duals
function linear_dual(A, b, lsolve, Af, T)

    # unpack dual numbers (if not dual numbers, since only one might be, just returns itself)
    bv = fd_value(b)

    # save factorization since we will perform two linear solves
    Afact = isnothing(Af) ? factorize(fd_value(A)) : Af

    # evaluate linear solver
    yv = lsolve(Afact, bv)

    # extract Partials of b - A * y  i.e., bdot - Adot * y  (since y does not contain duals)
    rhs = fd_partials(b - A*yv)

    # solve for new derivatives
    ydot = lsolve(Afact, rhs)

    # repack in ForwardDiff Dual
    return pack_dual(yv, ydot, T)
end


# Provide a ChainRule rule for reverse mode
function ChainRulesCore.rrule(::typeof(implicit_linear), A, b, lsolve, Af)

    # save factorization
    Afact = isnothing(Af) ? factorize(ReverseDiff.value(A)) : Af

    # evaluate solver
    y = lsolve(Afact, b)

    function implicit_pullback(ybar)
        u = lsolve(Afact', ybar)
        return NoTangent(), -u*y', u, NoTangent(), NoTangent()
    end

    return y, implicit_pullback
end

# register above rule for ReverseDiff
ReverseDiff.@grad_from_chainrules implicit_linear(A::Union{TrackedArray, AbstractArray{<:TrackedReal}}, b, lsolve, Af)
ReverseDiff.@grad_from_chainrules implicit_linear(A, b::Union{TrackedArray, AbstractArray{<:TrackedReal}}, lsolve, Af)
ReverseDiff.@grad_from_chainrules implicit_linear(A::Union{TrackedArray, AbstractArray{<:TrackedReal}}, b::Union{TrackedArray, AbstractVector{<:TrackedReal}}, lsolve, Af)


# function implicit_linear_inplace(A, b, y, Af)
#     Afact = isnothing(Af) ? A : Af
#     ldiv!(y, Afact, b)
# end

# function implicit_linear_inplace(A, b, y::AbstractVector{<:ForwardDiff.Dual{T}}, Af) where {T}

#     # unpack dual numbers (if not dual numbers, since only one might be, just returns itself)
#     bv = fd_value(b)
#     yv, ydot = unpack_dual(y)

#     # save factorization since we will perform two linear solves
#     Afact = isnothing(Af) ? factorize(fd_value(A)) : Af

#     # evaluate linear solver
#     ldiv!(yv, Afact, bv)

#     # extract Partials of b - A * y  i.e., bdot - Adot * y  (since y does not contain duals)
#     rhs = fd_partials(b - A*yv)

#     # solve for new derivatives
#     ldiv!(ydot, Afact, rhs)

#     # reassign y to this value
#     y .= pack_dual(yv, ydot, T)
# end


# ------ eigenvalues ------------

"""
    implicit_eigval(A, B, eigsolve)

Make eigenvalue problems AD compatible with ForwardDiff and ReverseDiff

# Arguments
- `A::matrix`, `B::matrix`: generlized eigenvalue problem. A v = λ B v  (B is identity for standard eigenvalue problem)
- `eigsolve::function`: λ, V, U = eigsolve(A, B). Function to solve the eigenvalue problem.  
    λ is a vector containing eigenvalues.  
    V is a matrix whose columns are the corresponding eigenvectors (i.e., λ[i] corresponds to V[:, i]).
    U is a matrix whose columns contain the left eigenvectors (u^H A = λ u^H B)
    The left eigenvectors must be in the same order as the right ones (i.e., U' * B * V must be diagonal).  
    U can be provided with any normalization as we normalize internally s.t. U' * B * V = I
    If A and B are symmetric/Hermitian then U = V.

# Returns
- `λ::vector`: eigenvalues and their derivatives.  (Currently only eigenvalue derivatives are provided.  not eigenvectors)
"""
implicit_eigval(A, B, eigsolve) = eigsolve(A, B)[1]  # If no AD, just solve normally. Returns only eigenvalue.

# forward cases
implicit_eigval(A::AbstractArray{<:ForwardDiff.Dual{T}}, B::AbstractArray{<:ForwardDiff.Dual{T}}, eigsolve) where {T} = eigval_fwd(A, B, eigsolve)
implicit_eigval(A, B::AbstractArray{<:ForwardDiff.Dual{T}}, eigsolve) where {T} = eigval_fwd(A, B, eigsolve)
implicit_eigval(A::AbstractArray{<:ForwardDiff.Dual{T}}, B, eigsolve) where {T} = eigval_fwd(A, B, eigsolve)

# reverse cases
implicit_eigval(A::Union{ReverseDiff.TrackedArray, AbstractArray{<:ReverseDiff.TrackedReal}}, B::Union{ReverseDiff.TrackedArray, AbstractArray{<:ReverseDiff.TrackedReal}}, eigsolve) where {T} = eigval_rev(A, B, eigsolve)
implicit_eigval(A, B::Union{ReverseDiff.TrackedArray, AbstractArray{<:ReverseDiff.TrackedReal}}, eigsolve) where {T} = eigval_rev(A, B, eigsolve)
implicit_eigval(A::Union{ReverseDiff.TrackedArray, AbstractArray{<:ReverseDiff.TrackedReal}}, B, eigsolve) where {T} = eigval_rev(A, B, eigsolve)

# extract values from A and B before passing to common function
eigval_fwd(A, B, eigsolve) = eigval_deriv(A, B, ForwardDiff.value.(A), ForwardDiff.value.(B), eigsolve)
eigval_rev(A, B, eigsolve) = eigval_deriv(A, B, ReverseDiff.value(A), ReverseDiff.value(B), eigsolve)

# eigenvalue derivatives used in both forward and reverse
function eigval_deriv(A, B, Av, Bv, eigsolve)
    λ, V, U = eigsolve(Av, Bv)
    U = U ./ (diag(U' * Bv * V))'  # normalize U s.t. U' * B * V = I
    
    # compute derivatives
    etype = promote_type(eltype(A), eltype(B))
    λd = similar(λ, complex(etype))
    for i = 1:length(λ)
        λd[i] = λ[i] + view(U, :, i)'*(A - λ[i]*B)*view(V, :, i)  # right hand side is zero for primal.  left side is zero for dual (only and A and B contain derivatives)
    end

    return λd
end


# -------------- non AD-able operations ------------------------

"""
    provide_rule(func, x, p=(); mode="ffd", jacobian=nothing, jvp=nothing, vjp=nothing)

Provide partials rather than rely on AD.  For cases where AD is not available
or to provide your own rule, or to use mixed mode, etc.

# Arguments
- `func::function`, `x::vector{float}`, `p::tuple`:  function of form: y = func(x, p), where x are variables and p are fixed parameters.
- `mode::string`:
    - "ffd": forward finite difference
    - "cfd": central finite difference
    - "cs": complex step
    - "jacobian": provide your own jacobian (see jacobian function)
    - "vp": provide jacobian vector product and vector jacobian product (see jvp and vjp)
- `jacobian::function`: only used if mode="jacobian". J = jacobian(x, p) provide J_ij = dy_i / dx_j
- `jvp::function`: only used if mode="vp" and in forward mode. ydot = jvp(x, p, v) provide Jacobian vector product J*v
- `vjp::function`: only used if mode="vp" and in revese mode. xbar = vjp(x, p, v) provide vector Jacobian product v'*J
"""
provide_rule(func, x, p=(); mode="ffd", jacobian=nothing, jvp=nothing, vjp=nothing) = provide_rule(func, x, p, mode, jacobian, jvp, vjp)

provide_rule(func, x, p, mode, jacobian, jvp, vjp) = func(x, p)

function provide_rule(func, x::AbstractVector{<:ForwardDiff.Dual{T}}, p, mode, jacobian, jvp, vjp) where {T}

    # unpack dual
    xv, xdot = unpack_dual(x)
    nx, nd = size(xdot)

    # call function with primal values
    yv = func(xv, p)
    ny = length(yv)

    # initialize
    ydot = Matrix{Float64}(undef, ny, nd)

    if mode == "ffd"  # forward finite diff

        h = sqrt(eps(Float64))  # theoretical optimal (absolute) step size

        # check whether we should do a series of JVPs or compute Jacobian than multiply

        if nd <= nx  # do JVPs (for each dual)
            xnew = Vector{Float64}(undef, nx)
            for i = 1:nd
                xnew .= xv + h*xdot[:, i]  # directional derivative
                ydot[:, i] = (func(xnew, p) - yv)/h
            end

        else  # compute Jacobian first
            J = Matrix{Float64}(undef, ny, nx)
            for i = 1:nx
                delta = max(xv[i]*h, h)  # use relative step size unless it is too small
                xv[i] += delta
                J[:, i] = (func(xv, p) - yv)/delta
                xv[i] -= delta
            end
            ydot .= J * xdot
        end

    elseif mode == "cfd"  # central finite diff

        h = cbrt(eps(Float64))  # theoretical optimal (absolute) step size

        # check whether we should do a series of JVPs or compute Jacobian than multiply

        if nd <= nx  # do JVPs
            xnew = Vector{Float64}(undef, nx)
            for i = 1:nd
                xnew .= xv + h*xdot[:, i]  # directional derivative
                yp = func(xnew, p)
                xnew .= xv - h*xdot[:, i]
                ym = func(xnew, p)
                ydot[:, i] = (yp - ym)/(2*h)
            end

        else  # compute Jacobian first
            J = Matrix{Float64}(undef, ny, nx)
            for i = 1:nx
                delta = max(xv[i]*h, h)  # use relative step size unless it is too small
                xv[i] += delta
                yp = func(xv, p)
                xv[i] -= 2*delta
                ym = func(xv, p)
                J[:, i] = (yp - ym)/(2*delta)
                xv[i] += delta  # reset
            end
            ydot .= J * xdot
        end

    elseif mode == "cs"  # complex step

        h = 1e-30
        xnew = Vector{ComplexF64}(undef, nx)

        # check whether we should do a series of JVPs or compute Jacobian than multiply

        if nd <= nx  # do JVPs (for each dual)
            for i = 1:nd
                xnew .= xv + h*im*xdot[:, i]  # directional derivative
                ydot[:, i] = imag(func(xnew, p))/h
            end

        else  # compute Jacobian first
            J = Matrix{Float64}(undef, ny, nx)
            xnew .= xv
            for i = 1:nx
                xnew[i] += h*im
                J[:, i] = imag(func(xnew, p))/h
                xnew[i] -= h*im
            end
            ydot .= J * xdot
        end

    elseif mode == "jacobian"
        J = jacobian(xv, p)
        ydot .= J * xdot

    elseif mode == "vp"  # jacobian vector product
        for i = 1:nd
            ydot[:, i] = jvp(xv, p, xdot[:, i])
        end

    else
        error("invalid mode")
    end

    return pack_dual(yv, ydot, T)
end


function ChainRulesCore.rrule(::typeof(provide_rule), func, x, p, mode, jacobian, jvp, vjp)

    # evaluate function
    y = func(x, p)
    nx = length(x)
    ny = length(y)

    if mode == "vp"

        function vppullback(ybar)
            xbar = vjp(x, p, ybar)
            return NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        return y, vppullback
    end

    J = Matrix{Float64}(undef, ny, nx)

    if mode == "ffd"

        h = sqrt(eps(Float64))
        for i = 1:nx
            delta = max(x[i]*h, h)  # use relative step size unless it is too small
            x[i] += delta
            J[:, i] = (func(x, p) - y)/delta
            x[i] -= delta
        end

    elseif mode == "cfd"

        h = cbrt(eps(Float64))
        for i = 1:nx
            delta = max(x[i]*h, h)  # use relative step size unless it is too small
            x[i] += delta
            yp = func(x, p)
            x[i] -= 2*delta
            ym = func(x, p)
            J[:, i] = (yp - ym)/(2*delta)
            x[i] += delta  # reset
        end

    elseif mode == "cs"

        h = 1e-30
        xnew = Vector{ComplexF64}(undef, nx)
        xnew .= x
        for i = 1:nx
            xnew[i] += h*im
            J[:, i] = imag(func(xnew, p))/h
            xnew[i] -= h*im
        end

    elseif mode == "jacobian"

        J .= jacobian(x, p)

    else
        error("invalid mode")
    end

    function pullback(ybar)
        xbar = J'*ybar
        return NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return y, pullback
end

ReverseDiff.@grad_from_chainrules provide_rule(func, x::TrackedArray, p, mode, jacobian, jvp, vjp)
ReverseDiff.@grad_from_chainrules provide_rule(func, x::AbstractArray{<:TrackedReal}, p, mode, jacobian, jvp, vjp)

end
