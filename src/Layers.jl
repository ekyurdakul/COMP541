using Knet
using MAT

#Read weights
function loadWeights(filename)
	#w=convert(Array{Float32,4}, w)
	#b=convert(Array{Float32,4}, tempb)
	return matread(filename)
end

#Dropout layer
@knet function dropout(x)
	return x .* rnd(init=Bernoulli(0.5, 2))
end

#VGGNet Conv Layer
@knet function VGGNetConv(x; inc=64, outc=64, winit=Gaussian(0.0, 0.01), binit=Constant(0.0), pad=1, wnd=3)
	w=par(init=winit, dims=(wnd,wnd,inc,outc))
	b=par(init=binit, dims=(1,1,outc))
	y=conv(w,x; window=wnd, padding=pad)
	return relu(y.+b)
end

#VGGNet Softmax
@knet function VGGNetSoftmax(x; winit=Gaussian(0.0, 0.01), binit=Constant(0.0))
	w=par(init=winit, dims=(1,1,4096,1000))
	b=par(init=binit, dims=(1,1,1000,1))
	y=conv(w,x; window=1)
	return soft(y.+b)
end

#Normal Softmax Layer
@knet function softmax(x; num=2, winit=Gaussian(0.0, 0.01), binit=Constant(0.0))
	w=par(init=winit, dims=(num, 0))
	b=par(init=binit, dims=(num, 1))
	return soft(w*x.+b)
end
