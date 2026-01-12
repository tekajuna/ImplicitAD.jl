# ---------- differentiable optimization problems----------------
implicit_opt(solve, objcon, P, opts, nx) = solve(P, opts)[1:nx] # Don't output Lagr.multipliers---or do?
"""
    implicit_opt(solve, objcon, P, opts,nx)
Make implicit function AD compatible (specifically with ForwardDiff and ReverseDiff).
# Arguments
- `solve::function`: xL = solve(P, opts). Solve optimization problem returning state variables `x` and `L`, for input variables `P`, and fixed optional parameters `opts`.
- `objcon::function`: 
- `P::vector{float}`: evaluation point; constant parameters given to the optimization problem
- `opts::tuple`: optional fixed parameters to solve. default is empty tuple. 
- `nx::int`: number of design variables, to distinguish output state from intermediate lagrange multipliers
- `method::string`: multiple methods are under development for the calculation of i/o derivatives. Options include {Hessian, residual}
"""



function implicit_opt(solve, objcon,θ_dual::AbstractVector{<:ForwardDiff.Dual{T}}, opts,nx::Int) where {T}
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
        @show length(xλ)
        # @show θv
        # @show x, l 
        lwrap(xλ) = lagrangian_eval(objcon,vcat(xλ,θv),nx,np,active)
        # A = ForwardDiff.hessian(lwrap,xλ)  
        start = Int(time_ns())
        A = ForwardDiff.jacobian(z̃->ReverseDiff.gradient(lwrap,z̃),xλ)
        A = sparse(A)
        println("Time for Hessian of Lagrangian: ", (time_ns()-start)*1e-9, " s")
        @show size(A)
        # eigvals_H = LinearAlgebra.eigvals(A)          # eigenvalues of Hessian of Lagrangian
        # condH     = cond(A)             # 2-norm condition number
        # println("Condition number: ", condH)
        # println("Smallest eigenvalue: ", minimum(abs.(eigvals_H)))
        # @show A
        start = Int(time_ns())
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
        println("Time for vector partials: ", (time_ns()-start)*1e-9, " s")
        @show size(b)
        # lwrap2(xλθ)= lagrangian_eval(objcon,vcat(xλθ),nx,np,active)
        # btest = ForwardDiff.hessian(lwrap2,[xλ;θv])[1:3,4:5] * θp
        # H =ForwardDiff.hessian(lwrap2,[xLv;θv])
        # @show A,size(A)
        # @show b
        @show cond(Array(A),2)

        start = Int(time_ns())
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
        Hxx_approx = ForwardDiff.hessian(x̃ -> lwrap([x̃;xλ[nx+1:end]]), xλ[1:nx])
        blocks = [inv(Hxx_approx + 1e-6I), I(nλ)]
        M_inv = blockdiag_precond(blocks)
        println("Time for diag extraction: ", (time_ns()-start)*1e-9, " s")
        @show cond(Matrix(M_inv*Array(A)),2)
        M_inv_op = make_preconditioner(blocks)
        start = Int(time_ns())

        # Aop = LinearOperator(Float64, nxl, nxl, true, false,
        #    (y, v) -> (y[:] = hvp(lwrap, xλ, v)))

        Aop = LinearOperator(Float64, nxl, nxl, true, false,
            (Y, V) -> begin
                if ndims(V) == 1
                    Y[:] = hvp(lwrap, xλ, V)
                else
                    Y .= hvp(lwrap, xλ, V)
                end
            end
        )
        # ydot = cg(Aop, b; abstol=1e-15,reltol=1e-15, maxiter=5000, log=false) #, history
        ydot = similar(b)
       
        for j in 1:size(b, 2)
            ydot[:, j] = -1.0 * cg(Aop, b[:, j];Pl=M_inv_op, abstol=1e-15,reltol=1e-15, maxiter=5000, log=false)
        end
        # ydot = -1.0 * (A\b)
        println("Time for mtrx-free linear solve: ", (time_ns()-start)*1e-9, " s")

        start = Int(time_ns())
        ydotBS = -1.0 * (A\b)
        println("Time for backslash linear solve: ", (time_ns()-start)*1e-9, " s")
        @show maximum(abs.(ydot - ydotBS))
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
function ChainRulesCore.rrule(::typeof(implicit_opt), solve, objcon,P,opts,NDV;method="residual")
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

ReverseDiff.@grad_from_chainrules implicit_opt(solve, objcon, P::ReverseDiff.TrackedArray, opts,NDV;method="residual")
ReverseDiff.@grad_from_chainrules implicit_opt(solve, objcon, P::AbstractVector{<:ReverseDiff.TrackedReal}, opts,NDV;method="residual")







#=
if method == "hvpold"
        x =  xLv[1:nx]
        np = length(Pv)
        λ0 =   xLv[nx+1:end]
        active = findall(>(1e-8),abs.(λ0))
        println("active Lagmults",active)
        xλ = vcat(x,λ0[active])
        nxl = length(xλ)
        lwrap(xλ) =   lagrangian_eval(objcon,vcat(xλ,Pv),nx,np,active)
        lwrap2(xlp) = lagrangian_eval(objcon,xlp,nx,np,active)
        # println("objcon eval",objcon(vcat(x,Pv)))
        # println("lagr eval", lagrangian_eval(objcon,vcat(xλ,Pv),nx,np,active))
        # println("ForwardDiff.derivative of Lagr wrt first DV: ",ForwardDiff.derivative(xE -> lagrangian_eval(objcon,vcat(xE,xλ[2:end],Pv),nx,np,active),x[1]))
        
        A = ForwardDiff.hessian(lwrap,xλ)
        xall = ForwardDiff.Dual.(vcat(xλ,Pv), ForwardDiff.Partials.(Tuple.(eachrow(vcat(zeros(nxl,size(Pd)[2]),Pd)))))
        Rcfg = ReverseDiff.GradientConfig(xall)
        b = fd_partials(ReverseDiff.gradient(lwrap2,xall,Rcfg))[1:nxl,:]
        println("A",A)
        println("b",b)
        # println(size(A))
        # println(size(b))
        # irrel = findall(<(1e-8),([sum(abs.(A[i,:])) for i=1:nx]))
        irrel  = findall(<(1e-8),([maximum(abs.(A[i,:])) for i=1:size(A,1)])) #size(A,1) but only for x, so 
                        # Choose based on solver tolerances 
        println("Irrells",irrel)
        # println("irrel",irrel)
        Ar = A[1:end .∉ [irrel],1:end .∉[irrel]] # remove inactive variables
        br= b[1:end .∉ [irrel],1:end ]
        # println("A",A)
        # println("b",b)

        println("Ar",Ar)
        println("br",br)
        println("cond(Ar)", cond(Ar))
        println("Max Ax-b svd\t", maximum(abs.(Ar*(svd(Ar)\br) - br)))
        println("Max Ax-b\t", maximum(abs.(Ar*(Ar\br) - br)))

        # println("sizes thereof, A and b",size(A),size(b))
        # println("sizes thereof, Ar and br",size(Ar),size(br))
        
        ytemp = -(svd(Ar)\br) 
        # println(,Ar * ytemp + br)
 
        if irrel[1] == 1
            println("Yep, it happened. Gonna need to add a case") # actually, should work great. 1:0 returns a zero-order matrix that we can vcat to
        end
        ydot = ytemp[1:irrel[1]-1,:] # Get the first couple of rows up until we hit the first irrel
        # println("Pd size",size(Pd))
        # println("IMPLICIT SIZES",size(ydot),size(zeros(1,np)))
        nzs = size(Pd)[2]
        ydot = vcat(ydot,zeros(1,nzs))
        if length(irrel) > 1
            for i =2:length(irrel)
                ydot = vcat(ydot,ytemp[irrel[i-1]-(i-2):irrel[i]-i,:])
                ydot = vcat(ydot, zeros(1,nzs))
            end
            ydot = vcat(ydot,ytemp[irrel[length(irrel)]-(length(irrel)-1):end,:])
        else
            ydot = vcat(ydot,ytemp[irrel[1]:end,:])
        end
        println("FINALTHING\t", A*ydot + b) 
        println("EXTREMA\t",extrema(A*ydot + b))

    end
    if method == "Hessian" # This method allows for a simpler residual function, but is inefficient with many inactive constraints or input parameters
        lagrangian=construct_lagrangian(objcon,xLv,Pv,nx)
        # H = ForwardDiff.hessian(lagrangian,vcat(xLv,Pv))
        # println("shape H", size(H))
        println("Using Hessian")
        # Initally, all variables are relevant, and we have indices for each DV and LM
        relevant = BitSet(1:length(xLv))
        # Remove indices of any lagrange multiplier that is zero
        for i=nx+1:length(xLv)
            if abs(xLv[i]) <1e-13
                delete!(relevant,i)
            end 
        end
        relevant = collect(relevant)
        println(relevant)
        xLvR = xLv[relevant] # Don't really use this, eh?

        # Extract Residual-DVs/multipliers derivatives from Hessian of Lagrangian
        # dRdxL = H[relevant,relevant] 
        dRdxL = ForwardDiff.jacobian(xtil -> ReverseDiff.gradient(xtilin-> lagrangian(vcat(xtilin,Pv)),xtil), xLv)[relevant,relevant]
        dRdP = ForwardDiff.jacobian(ptil -> ForwardDiff.gradient(xtil-> lagrangian(vcat(xtil,ptil)),xLv), Pv)[relevant,:]
        ytemp = -dRdxL\dRdP
        ydot = ytemp * Pd
        # return pack_dual(xLv[1:NDV], ydot[1:NDV,begin:end], T)
    end

    if method == "SHessian" # This method allows for a simpler residual function, but is inefficient with many inactive constraints or input parameters
        lagrangian=construct_lagrangian(objcon,xLv,Pv,nx)
        H = ForwardDiff.hessian(lagrangian,vcat(xLv,Pv))

        relevant = BitSet(1:length(xLv))
        # Remove indices of any lagrange multiplier that is zero
        for i=nx+1:length(xLv)
            if abs(xLv[i]) <1e-13
                delete!(relevant,i)
            end 
        end
        relevant = collect(relevant)
        # println(relevant)
        xLvR = xLv[relevant] # Don't really use this, eh?
        #eters derivatives from Hessian of Lagrangian
        
        if length(P)==1
            dRdP = H[relevant,end] # Each row is a residual equation; end col is parameter
        else
            dRdP = H[relevant,length(xLv)+1:end] #TODO: Fix this (What was need fix?)
        end 


        # Extract Residual-DVs/multipliers derivatives from Hessian of Lagrangian
        dRdxL = H[relevant,relevant] 
        ytemp = -dRdxL\dRdP
        ydot = ytemp * Pd
        # return pack_dual(xLv[1:NDV], ydot[1:NDV,begin:end], T)
    end
    if method == "Hessian2" # This method allows for a simpler residual function, but is inefficient with many inactive constraints or input parameters
        # Let's chop it down, first
        # Remove indices of any lagrange multiplier that is 0
        # println("HESS2")
        relcons = BitSet(1:length(objcon)) # indices of objective and constraints
        relvars = BitSet(1:length(xLv))
        for i = 2:length(relcons)
            if abs(xLv[nx+i-1]) < 1e-13
                delete!(relcons,i) # delete index to unused constraint
            end
        end
        for i=nx+1:length(xLv)
            if abs(xLv[i]) < 1e-13
                delete!(relvars,i)
            end
        end
        # Use a view
        relvars = collect(relvars)
        relcons = collect(relcons)

        
        # Construct relevant lagrangian
        lagrangian=construct_Rlagrangian(objcon,xLv[relvars],Pv,nx,relcons) # 

        H = ForwardDiff.hessian(lagrangian,vcat(xLv[relvars],Pv))
   
        
        dRdP = H[1:end-length(Pv),length(relvars)+1:end] 
 
        
        # Extract Residual-DVs/multipliers derivatives from Hessian of Lagrangian
        dRdxL = H[1:end-length(Pv),1:length(relvars)] # 

        ydot = -dRdxL\dRdP  
    end
    if method == "residual"
        # print("using residual")
        relevant = BitSet(1:length(xLv))

        # Remove indices of any lagrange multiplier that is 
        for i=nx+1:length(xLv)
            if abs(xLv[i]) <1e-13
                delete!(relevant,i)
            end
        end
        relevant = collect(relevant)
        xLvR = xLv[relevant]
        residual=construct_residuals(relevant,objcon,nx) # Is residuals already constructed? Has it changed in previous iteration?
        # solve for Jacobian-vector product

        #   jvp(residual, y, xd, p)
        b = jvp(residual, xLv, P, opts)
        
        # compute partial derivatives
        A = drdy_forward(residual, xLv, Pv, opts)[:,relevant]
        ydot = linear_solve(A, b)
    end

function construct_lagrangian(objcon,xLv,Pv,NDV)
    M = length(xLv) # Number of lagrange multipliers and DVs
    N = length(Pv)  # Number of input parameters
    function Lagrangian(X)
        Lagr = objcon[1](vcat(X[begin:NDV],X[M+1:end])) # functions are f(x,P)
        for i = 2:length(objcon) # for each L. Multiplier
            Lagr += objcon[i](vcat(X[begin:NDV],X[M+1:end]))* X[NDV+i-1]
        end
        return Lagr
    end
    return Lagrangian
end


function construct_Rlagrangian(objcon,xLv,Pv,NDV,relevant)
    M = length(xLv) # Number of lagrange multipliers and DVs
    N = length(Pv)  # Number of input parameters
    
    function Lagrangian(X)
        # Lagr = objcon[1](X[begin:NDV],X[M+1:end]) # functions are f(x,P)
        Lagr = objcon[1](vcat(X[begin:NDV],X[M+1:end]))
        # Lagr=0
        for i in relevant# for each L. Multiplier
            if i >1
                # Lagr += objcon[i](X[begin:NDV],X[M+1:end])* X[NDV+i-1]
                Lagr += objcon[i](vcat(X[begin:NDV],X[M+1:end]))* X[NDV+i-1]
            end
        end
        return Lagr
    end
    return Lagrangian
end

function construct_residuals(REL,objcon,NDV)
    function residual(xL,P,opts)
        # G =[]#type #change to preallocation, assign by index instead of push!
        G = Vector{Vector{ForwardDiff.Dual}}(undef,length(REL)-NDV +1)
        # push!(G,ForwardDiff.gradient(objcon[1],vcat(xL[begin:NDV],P))) 
        G[1] = ForwardDiff.gradient(objcon[1],vcat(xL[begin:NDV],P))
        # println("to,et1 ",typeof(G),eltype(G))
        for i = NDV+1:length(REL)
            # push!(G,xL[REL[i]]*ForwardDiff.gradient(objcon[REL[i]-NDV+1],vcat(xL[begin:NDV],P)))
            G[i-NDV+1] = xL[REL[i]]*ForwardDiff.gradient(objcon[REL[i]-NDV+1],vcat(xL[begin:NDV],P))
        end
        # println("to,et2 ",typeof(G),eltype(G))
        G = hcat(G...) # All the gradients 
        # println("to,et3 ",typeof(G),eltype(G))
        # println("typeo G: ", typeof(G[1]))
        # println("Val G ", G[1])
        
        # R = []#Array{Float64}(undef, NDV+length(REL))
        R = Vector{ForwardDiff.Dual}(undef,length(REL))
        for i = 1:NDV
            # push!(R,sum(G[i,:])) # Derivative of Lagr. wrt. ith DV
            R[i] = sum(G[i,:])
        end

        for i = NDV+1:length(REL)
            # push!(R, objcon[REL[i]-NDV+1](vcat(xL[begin:NDV],P)))
            R[i] =objcon[REL[i]-NDV+1](vcat(xL[begin:NDV],P))
        end
        # println("typeo R1: ", typeof(R))
        # println("Val R1 ", R[1],typeof(R[1]))

        R = vcat(R...)
        # println("typeo R2: ", typeof(R))
        # println("Val R2 ", R[1],typeof(R[1]))
        return R
        
    end 
    return residual
end


=#