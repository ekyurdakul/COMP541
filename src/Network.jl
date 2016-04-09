include("Models.jl")

@startTime("Compiling models...");
f2d=compile(:VGGNetFeature)
f3d=compile(:ORNFeature)
fClass=compile(:ORNClass)
@stopTime("Compilation completed.");


#Entire network architecture
@startTime("Loading network...")

function Network(x2d, x3d)
	#Compute feature vectors
	v2d=forw(f2d, x2d)
	v3d=forw(f3d, x3d)

	#Resize from [1,1,4096,1] to [4096,1]
	v2d=reshape(v2d, 4096, 1)

	#Predict class
	pClass=forw(fClass, v2d, v3d)

	println("Size of the output is: $(size(pClass))");
end

@stopTime("Loaded network.");
