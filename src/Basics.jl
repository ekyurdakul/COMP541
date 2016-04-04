using Knet

#Dropout layer
@knet function dropout(x)
	return x .* rnd(init=Bernoulli(1-pdrop, 1/(1-pdrop)))
end

#Save trained model weights to file
function saveTrainedModel(model, filename)
	JLD.save(filename, "model", clean(model))
end

#Load trained model
function loadTrainedModel(filename)
	return JLD.load(filename, "model")
end











#TBD: L1 Smooth layer
@knet function L1Smooth(x)
#=
	if (abs(x) < 1)
		y= 0.5*x*x;
	else
		y=abs(x)-0.5
	end
=#
end
