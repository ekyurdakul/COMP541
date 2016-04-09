include("Layers.jl")

weights=loadWeights("../data/julia_data/VGGNet.mat");

@knet function VGGNet(x)
	y=VGGNetConv(x; inc=3, outc=64, winit=weights["conv1_1_w"], binit=weights["conv1_1_b"])
	y=VGGNetConv(y; inc=64, outc=64, winit=weights["conv1_2_w"], binit=weights["conv1_2_b"])
	y=pool(y)

	y=VGGNetConv(y; inc=64, outc=128, winit=weights["conv2_1_w"], binit=weights["conv2_1_b"])
	y=VGGNetConv(y; inc=128, outc=128, winit=weights["conv2_2_w"], binit=weights["conv2_2_b"])
	y=pool(y)

	y=VGGNetConv(y; inc=128, outc=256, winit=weights["conv3_1_w"], binit=weights["conv3_1_b"])
	y=VGGNetConv(y; inc=256, outc=256, winit=weights["conv3_2_w"], binit=weights["conv3_2_b"])
	y=VGGNetConv(y; inc=256, outc=256, winit=weights["conv3_3_w"], binit=weights["conv3_3_b"])
	y=VGGNetConv(y; inc=256, outc=256, winit=weights["conv3_4_w"], binit=weights["conv3_4_b"])
	y=pool(y)

	y=VGGNetConv(y; inc=256, outc=512, winit=weights["conv4_1_w"], binit=weights["conv4_1_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=weights["conv4_2_w"], binit=weights["conv4_2_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=weights["conv4_3_w"], binit=weights["conv4_3_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=weights["conv4_4_w"], binit=weights["conv4_4_b"])
	y=pool(y)
	
	y=VGGNetConv(y; inc=512, outc=512, winit=weights["conv5_1_w"], binit=weights["conv5_1_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=weights["conv5_2_w"], binit=weights["conv5_2_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=weights["conv5_3_w"], binit=weights["conv5_3_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=weights["conv5_4_w"], binit=weights["conv5_4_b"])
	y=pool(y)

	#=
	y=VGGNetConv(y; inc=512, outc=4096, winit=w17, binit=b17, wnd=7, pad=0)
	y=dropout(y)

	y=VGGNetConv(y; inc=4096, outc=4096, winit=w18, binit=b18, wnd=1, pad=0)
	y=dropout(y)

	return VGGNetSoftmax(y; winit=w19, binit=b19)
	=#

	return y
end
