# ---------- differentiable optimization problems----------------
implicit_opt(solve, objcon, P, opts, NDV ;method="Hessian") = solve(P, opts)

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
        residual=construct_residuals(relevant,objcon,xLv,Pv,NDV)
        
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
function construct_residuals(REL,objcon,xLv,Pv,NDV)
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