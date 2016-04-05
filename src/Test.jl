include("Model.jl")

model=compile(:Model)
vgg=compile(:VGGNet)


x=ones(224,224,3,2)
y=forw(vgg,x)
size(y)



#=
batchsize=7;
x3d=zeros(30,30,30,3,batchsize);
x2d=zeros(4096,batchsize);

forw(model, x2d, x3d)

println("Model structure:")
Knet.netprint(model)
println()

println("Output sizes: (Batchsize=$batchsize)")
println("Class: $(size(get(model, :pClass)))")
println("Box:   $(size(get(model, :pBox)))")


=#
