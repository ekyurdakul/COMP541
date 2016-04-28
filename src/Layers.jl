#Macros to measure elapsed time
macro startTime(message)
	return quote
		tic()
		println($message)
	end
end
macro stopTime(message)
	return quote
		println($message)
		toc()
		println()
	end
end

@startTime("Importing packages...")
using Knet
using MAT
@stopTime("Import completed.")

@startTime("Loading weights...")
ORNWeights=matread("../data/julia_data/ORN.mat");
VGGWeights16=matread("../data/julia_data/VGG16.mat");
VGGWeights19=matread("../data/julia_data/VGG19.mat");
@stopTime("Loading completed.")

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

#Softmax Layer
@knet function softmax(x; num=2, winit=Gaussian(0.0, 0.01), binit=Constant(0.0))
	w=par(init=winit, dims=(num, 0))
	b=par(init=binit, dims=(num, 1))
	return soft(w*x.+b)
end
