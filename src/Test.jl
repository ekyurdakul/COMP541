include("Model.jl")

#model=compile(:Model);
model=compile(:VGGNet);

batchsize=1;
x3d=zeros(30,30,30,3,batchsize);
x2d=zeros(224,224,3,batchsize);

#y=forw(model, x2d, x3d);
y=forw(model, x2d)
Knet.netprint(model)
println("$(size(y))")
