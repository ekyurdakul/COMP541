using Knet

#Save trained model weights to file
function saveTrainedModel(model, filename)
	JLD.save(filename, "model", clean(model))
end

#Load trained model
function loadTrainedModel(filename)
	return JLD.load(filename, "model")
end
