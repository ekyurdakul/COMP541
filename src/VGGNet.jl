include("Layers.jl")

@knet function VGGNet(x)
	y=VGGNetConv(x; inc=3, outc=64)
	y=VGGNetConv(y; inc=64, outc=64)
	y=pool(y)

	y=VGGNetConv(y; inc=64, outc=128)
	y=VGGNetConv(y; inc=128, outc=128)
	y=pool(y)

	y=VGGNetConv(y; inc=128, outc=256)
	y=VGGNetConv(y; inc=256, outc=256)
	y=VGGNetConv(y; inc=256, outc=256)
	y=VGGNetConv(y; inc=256, outc=256)
	y=pool(y)

	y=VGGNetConv(y; inc=256, outc=512)
	y=VGGNetConv(y; inc=512, outc=512)
	y=VGGNetConv(y; inc=512, outc=512)
	y=VGGNetConv(y; inc=512, outc=512)
	y=pool(y)
	
	y=VGGNetConv(y; inc=512, outc=512)
	y=VGGNetConv(y; inc=512, outc=512)
	y=VGGNetConv(y; inc=512, outc=512)
	y=VGGNetConv(y; inc=512, outc=512)	
	y=pool(y)

	y=VGGNetFC(y; num=4096)
	y=VGGNetFC(y; num=4096)
	y=softmax(y; num=1000)

	return y
end
