
# ---------- differentiable optimization problems----------------
implicit_opt(solve, objcon, P, opts, NDV ;method="Hessian") = solve(P, opts)[1:NDV] # Don't output Lagr.multipliers

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
    Pd = fd_partials(P)
    if method == "hvp"
        println("LETS DO THE HESSIAN VECTOR PRODUCT")
        relevant = BitSet(1:length(xLv))
        # Remove indices of any lagrange multiplier that is 
        for i=NDV+1:length(xLv)
            if xLv[i] <1e-5
                delete!(relevant,i)
            end
        end
        relevant = collect(relevant)
        xLvR = xLv[relevant]
        objconR = objcon[relevant[NDV:end]]
        lagrangian=construct_lagrangian(objconR,xLvR,Pv,NDV)
        hvp = xtilde -> ForwardDiff.gradient(lagrangian,xtilde)(vcat(xLvR,Pv))[1]
        println(hvp)


    end
    if method == "Hessian" # This method allows for a simpler residual function, but is inefficient with many inactive constraints or input parameters
        println("HESS")
        @show Pd
        lagrangian=construct_lagrangian(objcon,xLv,Pv,NDV)
        H = ForwardDiff.hessian(lagrangian,vcat(xLv,Pv))
        # println("shape H", size(H))
        
        # Initally, all variables are relevant, and we have indices for each DV and LM
        relevant = BitSet(1:length(xLv))
        # Remove indices of any lagrange multiplier that is zero
        for i=NDV+1:length(xLv)
            if xLv[i] <1e-5
                delete!(relevant,i)
            end 
        end
        relevant = collect(relevant)
        xLvR = xLv[relevant] # Don't really use this, eh?
        # println("H ",H)
        # println("Rele ", relevant)
        # Extract Residual-InputParameters derivatives from Hessian of Lagrangian
        if length(P)==1
            dRdP = H[relevant,end] # Each row is a residual equation; end col is parameter
        else
            dRdP = H[relevant,length(xLv)+1:end] #TODO: Fix this (What was need fix?)
        end 

        # println("shape dRdP", size(dRdP))
        # println("dRdP ",dRdP)
        
        # Extract Residual-DVs/multipliers derivatives from Hessian of Lagrangian
        dRdxL = H[relevant,relevant] # 
        # println(dRdxL,"dRdxL")
        # println("shape dRdxL", size(dRdxL))
        # println("shape partials drdxl",size(fd_partials(dRdxL)))
        @show dRdxL dRdP size(dRdxL) size(dRdP)
        ytemp = -dRdxL\dRdP
        # @show  ydot
        ydot = ytemp * Pd
        # return pack_dual(xLv[1:NDV], ydot[1:NDV,begin:end], T)
    end
    if method == "Hessian2" # This method allows for a simpler residual function, but is inefficient with many inactive constraints or input parameters
        # Let's chop it down, first
        # Remove indices of any lagrange multiplier that is 0
        println("HESS2")
        relcons = BitSet(1:length(objcon)) # indices of objective and constraints
        relvars = BitSet(1:length(xLv))
        for i = 2:length(relcons)
            if xLv[NDV+i-1] <1e-5
                delete!(relcons,i) # delete index to unused constraint
            end
        end
        for i=NDV+1:length(xLv)
            if xLv[i] <1e-5
                delete!(relvars,i)
            end
        end
        relvars = collect(relvars)
        relcons = collect(relcons)
        println("relevants")
        println(relvars)
        println(relcons)
        # xLvR = xLv[relevant]
        
        # Construct relevant lagrangian
        lagrangian=construct_Rlagrangian(objcon,xLv[relvars],Pv,NDV,relcons) # 

        H = ForwardDiff.hessian(lagrangian,vcat(xLv[relvars],Pv))
        println("shape H", size(H))
        println("H ",H)
        # Initally, all variables are relevant, and we have indices for each DV and LM
        # relevant = BitSet(1:length(xLv))

        
        
        # Extract Residual-InputParameters derivatives from Hessian of Lagrangian
        # if length(P)==1
        #     dRdP = H[relevant,end]
        # else
        #     dRdP = H[relevant,[length(xLv)+1,end]] #TODO: Fix this
        # end 
        dRdP = H[1:end-length(Pv),[length(relvars)+1,end]] 
        println("shape dRdP", size(dRdP))
        println("dRdP ",dRdP)
        
        # Extract Residual-DVs/multipliers derivatives from Hessian of Lagrangian
        dRdxL = H[1:end-length(Pv),1:length(relvars)] # 
        println(dRdxL,"dRdxL")
        println("shape dRdxL", size(dRdxL))
        println("shape partials drdxl",size(fd_partials(dRdxL)))
        
        ydot = -dRdxL\dRdP  
    end
    if method == "residual"

        # Initally, all variables are relevant, and we have indices for each DV and LM
        relevant = BitSet(1:length(xLv))

        # Remove indices of any lagrange multiplier that is 
        for i=NDV+1:length(xLv)
            if xLv[i] <1e-13
                delete!(relevant,i)
            end
        end
        relevant = collect(relevant)
        # reconstruct output vector xLv to include only relevant pieces
        xLvR = xLv[relevant]
        residual=construct_residuals(relevant,objcon,NDV)
        # solve for Jacobian-vector product

        b = jvp(residual, xLv, P, opts)
        
        # compute partial derivatives
        A = drdy_forward(residual, xLv, Pv, opts)
        # linear solve

        ydot = linear_solve(A, b)
    end
    # println("\nALMOST THERE\n")
    # ANSWER = pack_dual(xLv[1:NDV], ydot[1:NDV,begin:end], T)
    # @show ANSWER
    # return ANSWER
    return pack_dual(xLv[1:NDV], ydot[1:NDV,begin:end], T) # return relevant partials
    # return pack_dual(xLvR[1:NDV], ydot[1:NDV,begin:end], T) # return relevant partials
end

function construct_lagrangian(objcon,xLv,Pv,NDV)
    M = length(xLv) # Number of lagrange multipliers and DVs
    N = length(Pv)  # Number of input parameters
    
    function Lagrangian(X)
        Lagr = objcon[1](X[begin:NDV],X[M+1:end]) # functions are f(x,P)
        for i = 2:length(objcon) # for each L. Multiplier
            Lagr += objcon[i](X[begin:NDV],X[M+1:end])* X[NDV+i-1]
        end
        return Lagr
    end

    return Lagrangian
    
end
function construct_Rlagrangian(objcon,xLv,Pv,NDV,relevant)
    M = length(xLv) # Number of lagrange multipliers and DVs
    N = length(Pv)  # Number of input parameters
    
    function Lagrangian(X)
        Lagr = objcon[1](X[begin:NDV],X[M+1:end]) # functions are f(x,P)
        # Lagr=0
        for i in relevant# for each L. Multiplier
            if i >1
                Lagr += objcon[i](X[begin:NDV],X[M+1:end])* X[NDV+i-1]
            end
        end
        return Lagr
    end

    return Lagrangian
    
end
function construct_residuals2(REL,objcon,NDV)
    function residual(xL,P,opts)

        R = []
        for i = 1 : NDV
            push!(R, ForwardDiff.gradient(objcon[1],vcat(xL[begin:NDV],P)))
        end
        for i = NDV+1 : length(xL)
            nothing
        end
        return R
    end
    return residual
end
function construct_residuals(REL,objcon,NDV)
    # M = length(xLv)
    # N = length(Pv)

    
    function residual(xL,P,opts)
        G =[]

        push!(G,ForwardDiff.gradient(objcon[1],vcat(xL[begin:NDV],P)))
        # @show NDV, size(objcon),size(P),size(REL),size(xL)
        # @show REL xL 
        for i = NDV+1:length(REL)
            # 9/21 changed first xL to xLv, should grab LagMult from full list using REL[i]
            # However, doing this causes the Problem1 to break. . . actually gets a matrix singularity
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

        R = vcat(R...)
        return R
        
    end 
    return residual
end

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

ReverseDiff.@grad_from_chainrules implicit_opt(solve, objcon, P::TrackedArray, opts,NDV;method="residual")
ReverseDiff.@grad_from_chainrules implicit_opt(solve, objcon, P::AbstractVector{<:TrackedReal}, opts,NDV;method="residual")

