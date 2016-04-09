include("Model.jl")

#x3d=zeros(30,30,30,3,batchsize);
x=zeros(224,224,3,1);
model=compile(:VGGNet);
y=forw(model, x)
#Knet.netprint(model)
println("$(size(y))")
