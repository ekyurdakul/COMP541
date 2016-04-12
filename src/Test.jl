include("Network.jl")

#Load actual classes
#y_real (20x1) vector containing num of classes for each class
y_real = matread("../data/julia_data/y_real.mat");
y_real = y_real["result"];

#y_pClass num of classes recognized by the network
y_pClass = zeros(20,1);

#batchsize=1 uses ~2GB GPU Memory
batchsize=1;
#number of scenes to process
maxscenes=5;

function classAccuracy(predicted, actual)
	
end

for i=1:maxscenes
	#Compute TSDF using CUDA
	run(`./tsdf $i`);

	#Read scene name
	tempfilename = open("..//data//julia_data//temp.txt");
	filename = readall(tempfilename);
	close(tempfilename);
	
	@startTime("Preparing 2D Input Data...");
	#Execute matlab script
	#run(`octave prepareScene.m $filename`);
	#tempmat=matread("../data/julia_data/temp.mat");
	@stopTime("Preparation completed.");
	
	@startTime("Loading input data...")
	#2D Input
	#boxcount = size(tempmat["input2d1"], 4);
	#boxcount += size(tempmat["input2d2"], 4);

	boxcount=2000;
	x2D=zeros(Float32,224,224,3,boxcount);
	#x2D[:,:,:, 1:500]=tempmat["input2d1"];
	#x2D[:,:,:, 501:boxcount]=tempmat["input2d2"];
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

	
	@startTime("Sending input to the network...")
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
		@startTime("Minibatching...");
		x2d=x2D[:,:,:, sx:ex];
		x3d=x3D[:,:,:,:, sx:ex];
		@stopTime("Minibatched.");

		@startTime("Sending minibatch to the network...");
		y_predict = Network(x2d, x3d);
		@startTime("Prediction completed.");

		for p=1:size(y_predict, 2)
			temp = zeros(20,1);
			temp[indmax(y_predict(:, p)), 1] = 1;
			temp[1,1] = 0; #Non objects are not relevant and arent in the data files so dont know
			y_pClass += temp;
		end

		#Print accuracy at the end of each batch
		acc=classAccuracy(y_pClass,y_real);
		println("Scene: $i Batch: $(convert(Int32,j)) Accuracy: $acc (%)");
	end
	@stopTime("Calculation completed.")
end
