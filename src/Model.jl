include("Basics.jl")

#Produces the top 4096 dimensional vector
@knet function Layer3D(x)
	w=par(init=Gaussian(0.0, 0.01), dims=(5,5,5,3,96))
	y=conv(w,x; window=5, padding=1, stride=1)
	y=relu(y)
	y=pool(y; window=2, padding=0, stride=2)

	w=par(init=Gaussian(0.0, 0.01), dims=(3,3,3,96,192))
	y=conv(w,y; window=3, stride=1)
	b=par(init=Constant(0.0), dims=(1,1,1,192,1))
	y=y.+b
	y=relu(y)
	y=pool(y; window=2, stride=2)

	w=par(init=Gaussian(0.0, 0.01), dims=(3,3,3,192,384))
	y=conv(w,y; window=3, stride=1)
	b=par(init=Constant(0.0), dims=(1,1,1,384,1))
	y=y.+b
	y=relu(y)

	w=par(init=Gaussian(0.0, 0.01), dims=(4096,0))
	b=par(init=Constant(0.0), dims=(4096,1))
	y=w*y.+b
	y=relu(y)
	y=dropout(y)
	return y
end

#TBD: ImageNet Produces the bottom 4096 dimensional vector
@knet function Layer2D(x)
	return x
end

#Class prediction
@knet function FCClass(x)
	w=par(init=Gaussian(0.0, 0.01), dims=(20,0))
	b=par(init=Constant(0.0), dims=(20,1))
	y=w*x.+b
	y=soft(y)
	return y
end

#TBD: Box prediction
@knet function FCBox(x)
	w=par(init=Gaussian(0.0, 0.01), dims=(120,0))
	b=par(init=Constant(0.0), dims=(120,1))
	y=w*x.+b
	#TBD:L1Smooth
	return y
end

#Entire model
@knet function Model(x2d, x3d)
	#Each produce 4096 dimensional vector
	v1=Layer3D(x3d)
	v2=Layer2D(x2d)

	#Reduce 8192 dimensions to 1000
	w1=par(init=Gaussian(0.0, 0.01), dims=(1000, 0))
	w2=par(init=Gaussian(0.0, 0.01), dims=(1000, 0))
	b1=par(init=Constant(0.0), dims=(1000,1))
	b2=par(init=Constant(0.0), dims=(1000,1))

	#Simulate concatenation
	y1=w1*v1.+b1
	y2=w2*v2.+b2
	v=y1+y2

	v=relu(v)
	v=dropout(v)
	
	#Feed v into 2 different layers, class and box prediction
	#pBox=FCBox(v)
	pClass=FCClass(v)
end
model=compile(:Model)








#Debug function
function printModel()
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
end
