include("Scene.jl")
using JLD
#julia> net2 = JLD.load("charlm.jld", "model")        # should create a copy of net

#Macros to measure elapsed time
macro startTime(message)
	return quote
		tic()
		println($message)
	end
end
macro stopTime(message)
	return quote
		println($message)
		toc()
		println()
	end
end
@startTime("Importing packages...")
using Knet
using MAT
@stopTime("Import completed.")

@startTime("Loading weights...")
VGGWeights16=matread("../data/julia_data/VGG16.mat");
VGGWeights19=matread("../data/julia_data/VGG19.mat");
@stopTime("Loading completed.")

#Dropout layer
@knet function dropout(x)
	return x .* rnd(init=Bernoulli(0.5, 2))
end
#Softmax Layer
@knet function softmax(x; num=2, winit=Gaussian(0.0, 0.01), binit=Constant(0.0))
	w=par(init=winit, dims=(num, 0))
	b=par(init=binit, dims=(num, 1))
	return soft(w*x.+b)
end
#VGGNet Conv Layer
@knet function VGGNetConv(x; inc=64, outc=64, winit=Gaussian(0.0, 0.01), binit=Constant(0.0), pad=1, wnd=3)
	w=par(init=winit, dims=(wnd,wnd,inc,outc))
	b=par(init=binit, dims=(1,1,outc))
	y=conv(w,x; window=wnd, padding=pad)
	return relu(y.+b)
end
#19 Layer VGGNet
#2D Feature Vector : Output size (1,1,4096,1)
@knet function VGGNet19(x)
	y=VGGNetConv(x; inc=3, outc=64, winit=VGGWeights19["conv1_1_w"], binit=VGGWeights19["conv1_1_b"])
	y=VGGNetConv(y; inc=64, outc=64, winit=VGGWeights19["conv1_2_w"], binit=VGGWeights19["conv1_2_b"])
	y=pool(y)

	y=VGGNetConv(y; inc=64, outc=128, winit=VGGWeights19["conv2_1_w"], binit=VGGWeights19["conv2_1_b"])
	y=VGGNetConv(y; inc=128, outc=128, winit=VGGWeights19["conv2_2_w"], binit=VGGWeights19["conv2_2_b"])
	y=pool(y)

	y=VGGNetConv(y; inc=128, outc=256, winit=VGGWeights19["conv3_1_w"], binit=VGGWeights19["conv3_1_b"])
	y=VGGNetConv(y; inc=256, outc=256, winit=VGGWeights19["conv3_2_w"], binit=VGGWeights19["conv3_2_b"])
	y=VGGNetConv(y; inc=256, outc=256, winit=VGGWeights19["conv3_3_w"], binit=VGGWeights19["conv3_3_b"])
	y=VGGNetConv(y; inc=256, outc=256, winit=VGGWeights19["conv3_4_w"], binit=VGGWeights19["conv3_4_b"])
	y=pool(y)

	y=VGGNetConv(y; inc=256, outc=512, winit=VGGWeights19["conv4_1_w"], binit=VGGWeights19["conv4_1_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights19["conv4_2_w"], binit=VGGWeights19["conv4_2_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights19["conv4_3_w"], binit=VGGWeights19["conv4_3_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights19["conv4_4_w"], binit=VGGWeights19["conv4_4_b"])
	y=pool(y)
	
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights19["conv5_1_w"], binit=VGGWeights19["conv5_1_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights19["conv5_2_w"], binit=VGGWeights19["conv5_2_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights19["conv5_3_w"], binit=VGGWeights19["conv5_3_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights19["conv5_4_w"], binit=VGGWeights19["conv5_4_b"])
	y=pool(y)

	#FC to generate 4096 feature vector
	return VGGNetConv(y; inc=512, outc=4096, winit=VGGWeights19["fc6_w"], binit=VGGWeights19["fc6_b"], wnd=7, pad=0)
end
#16 Layer VGGNet
#2D Feature Vector : Output size (1,1,4096,1)
@knet function VGGNet16(x)
	y=VGGNetConv(x; inc=3, outc=64, winit=VGGWeights16["conv1_1_w"], binit=VGGWeights16["conv1_1_b"])
	y=VGGNetConv(y; inc=64, outc=64, winit=VGGWeights16["conv1_2_w"], binit=VGGWeights16["conv1_2_b"])
	y=pool(y)

	y=VGGNetConv(y; inc=64, outc=128, winit=VGGWeights16["conv2_1_w"], binit=VGGWeights16["conv2_1_b"])
	y=VGGNetConv(y; inc=128, outc=128, winit=VGGWeights16["conv2_2_w"], binit=VGGWeights16["conv2_2_b"])
	y=pool(y)

	y=VGGNetConv(y; inc=128, outc=256, winit=VGGWeights16["conv3_1_w"], binit=VGGWeights16["conv3_1_b"])
	y=VGGNetConv(y; inc=256, outc=256, winit=VGGWeights16["conv3_2_w"], binit=VGGWeights16["conv3_2_b"])
	y=VGGNetConv(y; inc=256, outc=256, winit=VGGWeights16["conv3_3_w"], binit=VGGWeights16["conv3_3_b"])
	y=pool(y)

	y=VGGNetConv(y; inc=256, outc=512, winit=VGGWeights16["conv4_1_w"], binit=VGGWeights16["conv4_1_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights16["conv4_2_w"], binit=VGGWeights16["conv4_2_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights16["conv4_3_w"], binit=VGGWeights16["conv4_3_b"])
	y=pool(y)
	
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights16["conv5_1_w"], binit=VGGWeights16["conv5_1_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights16["conv5_2_w"], binit=VGGWeights16["conv5_2_b"])
	y=VGGNetConv(y; inc=512, outc=512, winit=VGGWeights16["conv5_3_w"], binit=VGGWeights16["conv5_3_b"])
	y=pool(y)

	#FC to generate 4096 feature vector
	return VGGNetConv(y; inc=512, outc=4096, winit=VGGWeights16["fc6_w"], binit=VGGWeights16["fc6_b"], wnd=7, pad=0)
end

#3D Feature Vector : Output size (4096,1)
@knet function ORNFeature(x)
	w=par(init=Gaussian(0.0, 0.01), dims=(5,5,5,3,96))
	b=par(init=Constant(0.0), dims=(1,1,1,96))
	y=conv(w,x; window=5, padding=1, stride=1)
	y=relu(y.+b)
	y=pool(y; window=2, padding=0, stride=2)

	w=par(init=Gaussian(0.0, 0.01), dims=(3,3,3,96,192))
	b=par(init=Constant(0.0), dims=(1,1,1,192))
	y=conv(w,y; window=3, stride=1)
	y=relu(y.+b)
	y=pool(y; window=2, stride=2)

	w=par(init=Gaussian(0.0, 0.01), dims=(3,3,3,192,384))
	b=par(init=Constant(0.0), dims=(1,1,1,384))
	y=conv(w,y; window=3, stride=1)
	y=relu(y.+b)

	w=par(init=Gaussian(0.0, 0.01), dims=(4096,24576))
	b=par(init=Constant(0.0), dims=(4096,1))
	y=relu(w*y.+b)

	return dropout(y)
end

@knet function ORNClass(v2d, v3d)
	#Simulate concatenation and reduce dimension to 1000
	w=par(init=Gaussian(0.0, 0.01), dims=(1000, 4096))
	b=par(init=Constant(0.0), dims=(1000,1))

	y2d=w*v2d.+b
	y3d=w*v3d.+b
	v=relu(y3d+y2d)
	v=dropout(v)

	#Classify
	return softmax(v; num=20)
end

#Entire network architecture
@knet function Network(x2d, x3d)
	y1 = ORNFeature(x3d);
	y2 = ORNClass(x2d, y1);
	return y2;
end
network=compile(:Network);




#Choose VGGNet type, default to the paper's version
vggtype = parse(Int32, ARGS[2]);
if vggtype != 16 && vggtype != 19
	vggtype = 16;
end

#Choose model according to argument, free the other version's loaded weights
if vggtype == 16
	f2d=compile(:VGGNet16)
	VGGWeights19 = 0;
elseif vggtype == 19
	f2d=compile(:VGGNet19)
	VGGWeights16 = 0;
end

#Training options
setp(network; lr=0.01);
#setp(network; momentum = 0.9);
setp(network; l2reg = 0.0005);
#epochs = 10000;

#y
hot_vec = matread("../data/julia_data/hot_train.mat");
hot_vec = hot_vec["hot"];
hot_vec = convert(Array{Float32,3}, hot_vec);


batchsize=20;
sumloss = 0;
sloss = 0;
numloss = 0;

maxscenes = parse(Int32, ARGS[1]);
if maxscenes < 1
	maxscenes = 1;
elseif maxscenes > 795
	maxscenes = 795
end

epochs = parse(Int32, ARGS[3]);
if epochs <= 0
	epochs = 1;
end

println("Number of scenes to be processed is: $maxscenes\n");
println("***Start time: $(now())***");
@startTime("***Training for $epochs epochs...***\n");
for epoch=1:epochs

	if epoch == 5000
		setp(network; lr = 0.001);
	elseif epoch == 10000
		setp(network; lr = 0.0001);
	end
	
	for i=1:maxscenes
		@startTime("Preparing 3D Input Data...");
		#Compute TSDF using CUDA
		run(`./tsdf_train $i`);

		#Read scene name
		tempfilename = open("..//data//julia_data//temp.txt");
		filename = readall(tempfilename);
		close(tempfilename);
		@stopTime("Preparation completed.");

		@startTime("Preparing 2D Input Data...");
		x2D, boxcount = prepareScene(filename);
		@stopTime("Preparation completed.");

		@startTime("Loading input data...")
		#2D Input
		x2D=convert(Array{Float32,4}, x2D);
		#3D Input
		TSDFfile=open("../data/julia_data/temp.tdsf", "r");
		x3D=zeros(Float32, boxcount, 3, 30, 30, 30);
		read!(TSDFfile, x3D);
		close(TSDFfile);
		x3D=permutedims(x3D, [5 4 3 2 1]);
		#Minibatching
		batchcount=floor(boxcount/batchsize);
		if boxcount%batchsize!=0
			batchcount+=1
		end
		@stopTime("Loaded input data.")


		@startTime("Training...")
		for j=1:batchcount
			sx=1+(j-1)*batchsize;
			ex=batchsize*j;

			#Handle special case
			if boxcount%batchsize!=0 && j==batchcount
				ex=boxcount;
			end
			sx=convert(Int32, sx);
			ex=convert(Int32, ex);

			#Feed the network with minibatches
			x2d=x2D[:,:,:, sx:ex];
			x3d=x3D[:,:,:,:, sx:ex];

			x2d = forw(f2d, x2d);
			x2d = reshape(x2d, size(x2d));

			#Train
			y_predict = forw(network, x2d, x3d);
			back(network, y_predict, softloss);
			update!(network);

			sumloss += zeroone(y_predict, hot_vec[:, sx:ex, i]);
			sloss += softloss(y_predict, hot_vec[:, sx:ex, i]);
			numloss += 1;

			loss = sumloss/numloss;

			#Print accuracy at the end of each batch
			println("Epoch: $epoch Scene: $i Batch: $(convert(Int32,j)) Softloss: $(sloss/numloss) Accuracy: $((1-loss)*100)%");
		end
		@stopTime("Calculation completed.")
	end

	@startTime("Saving computed weights...")
	JLD.save("..//experiments//trainedModel_vgg_$(vggtype)_$(maxscenes)_scenes_$(epochs)_epochs.jld", "model", clean(network));
	@stopTime("Saving completed.")
end
@stopTime("***Training completed.***");
println("***End time: $(now())***");
