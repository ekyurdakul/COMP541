using Knet

#Dropout layer
@knet function dropout(x)
	return x .* rnd(init=Bernoulli(0.5, 2))
end

#VGGNet Conv Layer
@knet function VGGNetConv(x; inc=64, outc=64, winit=Gaussian(0.0, 0.01), binit=Constant(0.0), pad=1, wnd=3)
	w=par(init=winit, dims=(wnd,wnd,inc,outc))
	b=par(init=binit, dims=(1,1,outc,1))
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












#TBD: L1 Smooth layer
#=
@knet function L1Smooth(x)
	if (abs(x) < 1)
		y= 0.5*x*x;
	else
		y=abs(x)-0.5
	end
end
=#
