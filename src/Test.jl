include("Model.jl")



model=compile(:Model);


batchsize=1;
x3d=zeros(30,30,30,3,batchsize);
x2d=zeros(50,50,3,batchsize);

#y=forw(model, x2d, x3d);









#=
println("Model structure:")
Knet.netprint(model)
println()
println("Output sizes: (Batchsize=$batchsize)")
println("Class: $(size(get(model, :pClass)))")
=#
