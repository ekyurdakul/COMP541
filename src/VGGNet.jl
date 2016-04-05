include("Basics.jl")
include("Layers.jl")

#Load weights
w1,b1=loadWeights("../data/conv1_1.mat");
w2,b2=loadWeights("../data/conv1_2.mat");
w3,b3=loadWeights("../data/conv2_1.mat");
w4,b4=loadWeights("../data/conv2_2.mat");
w5,b5=loadWeights("../data/conv3_1.mat");
w6,b6=loadWeights("../data/conv3_2.mat");
w7,b7=loadWeights("../data/conv3_3.mat");
w8,b8=loadWeights("../data/conv3_4.mat");
w9,b9=loadWeights("../data/conv4_1.mat");
w10,b10=loadWeights("../data/conv4_2.mat");
w11,b11=loadWeights("../data/conv4_3.mat");
w12,b12=loadWeights("../data/conv4_4.mat");
w13,b13=loadWeights("../data/conv5_1.mat");
w14,b14=loadWeights("../data/conv5_2.mat");
w15,b15=loadWeights("../data/conv5_3.mat");
w16,b16=loadWeights("../data/conv5_4.mat");
w17,b17=loadWeights("../data/fc6.mat");
w18,b18=loadWeights("../data/fc7.mat");
w19,b19=loadWeights("../data/fc8.mat");

@knet function VGGNet(x)
	y=VGGNetConv(x; inc=3, outc=64, winit=w1, binit=b1)
	y=VGGNetConv(y; inc=64, outc=64, winit=w2, binit=b2)
	y=pool(y)

	y=VGGNetConv(y; inc=64, outc=128, winit=w3, binit=b3)
	y=VGGNetConv(y; inc=128, outc=128, winit=w4, binit=b4)
	y=pool(y)

	y=VGGNetConv(y; inc=128, outc=256, winit=w5, binit=b5)
	y=VGGNetConv(y; inc=256, outc=256, winit=w6, binit=b6)
	y=VGGNetConv(y; inc=256, outc=256, winit=w7, binit=b7)
	y=VGGNetConv(y; inc=256, outc=256, winit=w8, binit=b8)
	y=pool(y)

	y=VGGNetConv(y; inc=256, outc=512, winit=w9, binit=b9)
	y=VGGNetConv(y; inc=512, outc=512, winit=w10, binit=b10)
	y=VGGNetConv(y; inc=512, outc=512, winit=w11, binit=b11)
	y=VGGNetConv(y; inc=512, outc=512, winit=w12, binit=b12)
	y=pool(y)
	
	y=VGGNetConv(y; inc=512, outc=512, winit=w13, binit=b13)
	y=VGGNetConv(y; inc=512, outc=512, winit=w14, binit=b14)
	y=VGGNetConv(y; inc=512, outc=512, winit=w15, binit=b15)
	y=VGGNetConv(y; inc=512, outc=512, winit=w16, binit=b16)
	y=pool(y)

	y=VGGNetConv(y; inc=512, outc=4096, winit=w17, binit=b17, wnd=7, pad=0)
	y=dropout(y)

	y=VGGNetConv(y; inc=4096, outc=4096, winit=w18, binit=b18, wnd=1, pad=0)
	y=dropout(y)

	return VGGNetSoftmax(y; winit=w19, binit=b19)
end
