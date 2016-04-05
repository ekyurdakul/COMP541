include("Layers.jl")

@knet function VGGNet(x)
	y=VGGNetConv(x, 3, 64)
	y=VGGNetConv(y, 64, 64)
	y=pool(y)

	y=VGGNetConv(y, 64, 128)
	y=VGGNetConv(y, 128, 128)
	y=pool(y)

	y=VGGNetConv(y, 128, 256)
	y=VGGNetConv(y, 256, 256)
	y=VGGNetConv(y, 256, 256)
	y=VGGNetConv(y, 256, 256)
	y=pool(y)

	y=VGGNetConv(y, 256, 512)
	y=VGGNetConv(y, 512, 512)
	y=VGGNetConv(y, 512, 512)
	y=VGGNetConv(y, 512, 512)
	y=pool(y)
	
	y=VGGNetConv(y, 512, 512)
	y=VGGNetConv(y, 512, 512)
	y=VGGNetConv(y, 512, 512)
	y=VGGNetConv(y, 512, 512)	
	y=pool(y)

	y=VGGNetFC(y, 4096)
	y=VGGNetFC(y, 4096)
	y=softmax(y, 1000)

	return y
end
