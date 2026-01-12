# ---------- differentiable optimization problems----------------
implicit_kkt(solve, objcon, P, opts, nx) = solve(P, opts)[1:nx] # Don't output Lagr.multipliers---or do?
"""
    implicit_kkt(solve, objcon, P, opts,nx)
Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).
# Arguments
- `solve::function`: xL = solve(P, opts). Solve optimization problem returning state variables `x` and `L`, for input variables `P`, and fixed optional parameters `opts`.
- `objcon::function`: 
- `P::vector{float}`: evaluation point; constant parameters given to the optimization problem
- `opts::tuple`: optional fixed parameters to solve. default is empty tuple. 
- `nx::int`: number of design variables, to distinguish output state from intermediate lagrange multipliers
- `method::string`: multiple methods are under development for the calculation of i/o derivatives. Options include {Hessian, residual}
"""



function implicit_kkt(solve, objcon,θ_dual::AbstractVector{<:ForwardDiff.Dual{T}}, opts,nx::Int) where {T}
    θv = fd_value(θ_dual)        # get parameter values, isolated from derivatives
    np = length(θv)
    θp = fd_partials(θ_dual)
    # println(opts)
    xLv = solve(θv,opts)          # solve for x and λ, given input parameters
    # @show xLv
    # objcon(x̃) = objcon(x̃,opts)
    if length(xLv)== nx    # Unconstrained Case 
        # println("Here we are unconstrained!")
        x = xLv
        # @show x, θv
        A = ForwardDiff.hessian(x̃->objcon([x̃;θv]),x) # Possible issue if Objcon returns a 1-vector, Fine if returns a real
        # @show A
        b = fd_partials(ForwardDiff.gradient(x̃->objcon([x̃;θ_dual]),x))
        # @show b
        # A = svd(A)
        ydot = -A\b
    else
        x = @view xLv[1:nx]       # All DVs
        l = @view xLv[nx+1:end]   # All Lambdas
        # active = findall(>(-1e-7),objcon([x;θv])[2:end])
        # @show x, l
        # @show abs.(objcon([x;θv])[2:end])
        active = findall(<(1e-5),abs.(objcon([x;θv])[2:end])) 
        # @show objcon([x;θv])
        # @show active
        l_active = l[active] 
        xλ=[x;l_active]
        nxl = length(xλ)
        nλ = length(active)
        # @show length(xλ)
        # @show θv
        # @show x, l 
        lwrap(xλ) = lagrangian_eval(objcon,vcat(xλ,θv),nx,np,active)
        # A = ForwardDiff.hessian(lwrap,xλ)  
        # start = Int(time_ns())
        A = ForwardDiff.jacobian(z̃->ReverseDiff.gradient(lwrap,z̃),xλ)
        A = sparse(A)
        # println("Time for Hessian of Lagrangian: ", (time_ns()-start)*1e-9, " s")
        # @show size(A)
        # eigvals_H = LinearAlgebra.eigvals(A)          # eigenvalues of Hessian of Lagrangian
        # condH     = cond(A)             # 2-norm condition number
        # println("Condition number: ", condH)
        # println("Smallest eigenvalue: ", minimum(abs.(eigvals_H)))
        # @show A
        # start = Int(time_ns())
        # R2(t) = ForwardDiff.gradient(x̃λ->lagrangian_eval(objcon,[x̃λ;t],nx,np,active),xλ)
        # b = fd_partials(R2([θ_dual;]))
        R2(t) = ForwardDiff.gradient(x̃->lagrangian_eval(objcon,[x̃;l_active;t],nx,np,active),x)
        bx = fd_partials(R2(θ_dual))
        if !isempty(active) # constraints active
            bl = fd_partials(objcon([x;θ_dual])[active .+ 1])
            b = vcat(bx,bl)
        else # constraints inactive
            b = bx
        end
        # println("Time for vector partials: ", (time_ns()-start)*1e-9, " s")
        # @show size(b)
        # lwrap2(xλθ)= lagrangian_eval(objcon,vcat(xλθ),nx,np,active)
        # btest = ForwardDiff.hessian(lwrap2,[xλ;θv])[1:3,4:5] * θp
        # H =ForwardDiff.hessian(lwrap2,[xLv;θv])
        # @show A,size(A)
        # @show b
        # @show cond(Array(A),2)

        # start = Int(time_ns())
        # diagH = zeros(nxl)
        # for i in 1:nxl
        #     eᵢ = zeros(nxl); eᵢ[i] = 1.0
        #     diagH[i] = dot(eᵢ, hvp(lwrap, xλ, eᵢ))
        # end

        # M_inv = Diagonal(1.0 ./ (diagH .+ 1e-12))
        # α = 1e-3  # or a small estimate of average Hessian magnitude
        # M_inv = Diagonal(fill(1/α, nxl))
        # nx = ...   # dimension of primal variables
        # nλ = ...   # number of constraints
        # Hxx_approx = ForwardDiff.hessian(x̃ -> lwrap([x̃;xλ[nx+1:end]]), xλ[1:nx])
        # blocks = [inv(Hxx_approx + 1e-6I), I(nλ)]
        # M_inv = blockdiag_precond(blocks)
        # println("Time for diag extraction: ", (time_ns()-start)*1e-9, " s")
        # @show cond(Matrix(M_inv*Array(A)),2)
        # M_inv_op = make_preconditioner(blocks)
        # start = Int(time_ns())

        # Aop = LinearOperator(Float64, nxl, nxl, true, false,
        #    (y, v) -> (y[:] = hvp(lwrap, xλ, v)))

        # Aop = LinearOperator(Float64, nxl, nxl, true, false,
        #     (Y, V) -> begin
        #         if ndims(V) == 1
        #             Y[:] = hvp(lwrap, xλ, V)
        #         else
        #             Y .= hvp(lwrap, xλ, V)
        #         end
        #     end
        # )
        # ydot = cg(Aop, b; abstol=1e-15,reltol=1e-15, maxiter=5000, log=false) #, history
        # ydot = similar(b)
       
        # for j in 1:size(b, 2)
        #     ydot[:, j] = -1.0 * cg(Aop, b[:, j];Pl=M_inv_op, abstol=1e-15,reltol=1e-15, maxiter=5000, log=false)
        # end
        # ydot = -1.0 * (A\b)
        # println("Time for mtrx-free linear solve: ", (time_ns()-start)*1e-9, " s")

        # start = Int(time_ns())
        ydot = -1.0 * (A\b)
        # println("Time for backslash linear solve: ", (time_ns()-start)*1e-9, " s")
        # @show maximum(abs.(ydot - ydotBS))
        # @show ydot
        # if sum(ydot)==0.0
        #     println("WEIRD!")
        #     @show solve, objcon,θ_dual, opts,nx
        # end
    end
    return pack_dual(x, ydot[1:nx,begin:end], T) # return relevant partials
end

function lagrangian_eval(objcon, X::AbstractVector{T}, nx::Int, nθ::Int, active::Vector{Int}) where T
    # X = [x... λ... θ...]
    xp = X[[1:nx; end-nθ+1:end]]          # primal variables + parameters
    λ  = X[nx+1:end-nθ]                   # Lagrange multipliers (active only)
    oc = objcon(xp)                       # objective + constraints
    # pick objective + active constraints
    oc_active = oc[vcat(1, 1 .+ active)]  # 1 = objective, 1.+active = active constraints
    L = oc_active[1]                       # start with objective
    for i in 1:length(λ)
        L += λ[i] * oc_active[i+1]        # add active constraints weighted by λ
    end
    return L
end

function hvp(f, t, V)
    g(t) = ReverseDiff.gradient(f, t)
    if ndims(V) == 1
        # single vector case
        return ForwardDiff.derivative(ε -> g(t .+ ε .* V), 0.0)
    else
        # matrix case: apply columnwise
        n, m = size(V)
        Y = similar(V)
        for j in 1:m
            Y[:, j] = ForwardDiff.derivative(ε -> g(t .+ ε .* V[:, j]), 0.0)
        end
        return Y
    end
end

function blockdiag_precond(blocks)
    n = sum(size(B, 1) for B in blocks)
    function mv!(y, x)
        idx = 1
        for B in blocks
            nB = size(B, 1)
            y[idx:idx+nB-1] .= B \ x[idx:idx+nB-1]
            idx += nB
        end
        return y
    end
    return LinearOperator(Float64, n, n, true, true, mv!)
end

function make_preconditioner(blocks)
    M = blockdiag_precond(blocks)  # or however you build your block system
    LinearOperator(Float64, size(M,1), size(M,2);
                   prod = (y, x) -> mul!(y, M, x),      # y .= M * x
                   ldiv = (y, x) -> ldiv!(y, M, x))     # y .= M \ x
end


# ReverseDiff (Not likely to be used as a good GP problem will involve few inputs and many outputs)
function ChainRulesCore.rrule(::typeof(implicit_kkt), solve, objcon,P,opts,NDV;method="residual")
    xLv = copy(solve(P,opts)) # Get outputs of solve: DVs and Lags
    Pv = ReverseDiff.value(P) # Value of input parameters stripped from dual
    relevant = BitSet(1:length(xLv))
    # Remove indices of any lagrange multiplier that is 
    for i=NDV+1:length(xLv)
        if xLv[i] <1e-5
            delete!(relevant,i)
        end
    end
    relevant = collect(relevant)

    # reconstruct output vector xLv to include only relevant pieces
    xLvR = xLv[relevant] # Only relevant multipliers and values
    if method == "residual"
        residual = construct_residuals(relevant,objcon,xLv,Pv,NDV)
        function pullback(ybar)
            #Construct residuals
            A = drdy_forward(residual, xLvR,P,opts)
            u = linear_solve(A',ybar)
            xbar = vjp(residual,xLvR,P,opts,-u)
            return NoTangent(), NoTangent(),NoTangent(), xbar, NoTangent(), NoTangent()
        end
    end
    if method == "VHP"
        nothing
    end

    return xLvR, pullback
end

ReverseDiff.@grad_from_chainrules implicit_kkt(solve, objcon, P::ReverseDiff.TrackedArray, opts,NDV;method="residual")
ReverseDiff.@grad_from_chainrules implicit_kkt(solve, objcon, P::AbstractVector{<:ReverseDiff.TrackedReal}, opts,NDV;method="residual")




const implicit_opt=implicit_kkt


