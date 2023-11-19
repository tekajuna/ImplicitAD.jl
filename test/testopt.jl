using ImplicitAD
using PyCall
using SNOW
using Snopt

m = Model(Ipopt.Optimizer)
@variable(m,x )
@variable(m,y>=0)

@NLobjective(m,Min,(1-x)^2 + 100(y-x^2)^2)

@NLconstraint(m,c1, y>= 2*x^2)

optimize!(m)

println(value(x))
println(value(y))
println(objective_value(m))
println(dual(c1))
println(value(c1))