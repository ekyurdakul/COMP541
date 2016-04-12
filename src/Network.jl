include("Models.jl")

f2d=compile(:VGGNetFeature)
f3d=compile(:ORNFeature)
fClass=compile(:ORNClass)


#Entire network architecture
function Network(x2d, x3d)
	#Compute feature vectors
	v2d=forw(f2d, x2d)
	v3d=forw(f3d, x3d)

	#Resize from [1,1,4096,1] to [4096,1]
	v2d=reshape(v2d, size(v2d));

	#Predict class
	pClass=forw(fClass, v2d, v3d)

	println("Size of the output is: $(size(pClass))");
end
