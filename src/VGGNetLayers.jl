using Knet

@knet function conv3(x, inc, outc)
	w=par(init=Gaussian(0.0, 0.01), dims=(3,3,inc,outc))
	y=conv(w,x; window=3, padding=1)
	y=relu(y)
	return y
end

@knet function fc(x, num)
	w=par(init=Gaussian(0.0, 0.01), dims=(num, 0))
	b=par(init=Constant(0.0), dims=(num, 1))
	y=w*x.+b
	y=relu(y)
	y=dropout(y)
	return y
end

@knet function softmax(x, num)
	w=par(init=Gaussian(0.0, 0.01), dims=(num, 0))
	b=par(init=Constant(0.0), dims=(num, 1))
	y=w*x.+b
	y=soft(y)
	return y
end
