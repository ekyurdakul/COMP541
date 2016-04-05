using Knet
using HDF5

#Save trained model weights to file
function saveTrainedModel(model, filename)
	JLD.save(filename, "model", clean(model))
end

#Load trained model
function loadTrainedModel(filename)
	return JLD.load(filename, "model")
end

#Read weights
function loadWeights(filename)
	w=h5read(filename, "w")
	b=h5read(filename, "b")
	w=w["value"]
	b=b["value"]
	tempb=zeros(1,1,size(b,1), size(b,2))
	tempb[1,1,:,:]=b
	w=convert(Array{Float64,4}, w)
	b=convert(Array{Float64,4}, tempb)
	return w,b
end
