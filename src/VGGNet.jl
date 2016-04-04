include("VGGNetLayers.jl")

@knet function VGGNet(x)
	y=conv3(x, 3, 64)
	y=conv3(y, 64, 64)
	y=pool(y)

	y=conv3(y, 64, 128)
	y=conv3(y, 128, 128)
	y=pool(y)

	y=conv3(y, 128, 256)
	y=conv3(y, 256, 256)
	y=conv3(y, 256, 256)
	y=conv3(y, 256, 256)
	y=pool(y)

	y=conv3(y, 256, 512)
	y=conv3(y, 512, 512)
	y=conv3(y, 512, 512)
	y=conv3(y, 512, 512)
	y=pool(y)
	
	y=conv3(y, 512, 512)
	y=conv3(y, 512, 512)
	y=conv3(y, 512, 512)
	y=conv3(y, 512, 512)	
	y=pool(y)

	y=fc(y, 4096)
	y=fc(y, 4096)
	y=softmax(y, 1000)

	return y
end

model=compile(:VGGNet)
x=ones(224,224,3,2)

y=forw(model,x)

size(y)
