using Knet

#Dropout layer
@knet function dropout(x)
	return x .* rnd(init=Bernoulli(0.5, 2))
end

#VGGNet Conv Layer
@knet function VGGNetConv(x, inc, outc)
	w=par(init=Gaussian(0.0, 0.01), dims=(3,3,inc,outc))
	y=conv(w,x; window=3, padding=1)
	y=relu(y)
	return y
end

#VGGNet FC Layer
@knet function VGGNetFC(x, num)
	w=par(init=Gaussian(0.0, 0.01), dims=(num, 0))
	b=par(init=Constant(0.0), dims=(num, 1))
	y=w*x.+b
	y=relu(y)
	y=dropout(y)
	return y
end

#Softmax Layer
@knet function softmax(x, num)
	w=par(init=Gaussian(0.0, 0.01), dims=(num, 0))
	b=par(init=Constant(0.0), dims=(num, 1))
	y=w*x.+b
	y=soft(y)
	return y
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
