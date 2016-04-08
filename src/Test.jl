include("Model.jl")

x=zeros(224,224,3,1);
x=convert(Array{Float32,4}, x);

model=compile(:VGGNet);
y=forw(model, x)

Knet.netprint(model)
println("$(size(y))")

#x3d=zeros(30,30,30,3,batchsize);
