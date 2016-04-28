include("Models.jl")
include("Scene.jl")

#Load actual results from the paper
#20x2000x654 matrix containing 1 hot vectors for all bounding boxes in all files
#20 classes 2000 bounding boxes 654 files
hot_vec = matread("../data/julia_data/hot.mat");
hot_vec = hot_vec["hot"];

#Comparing my systems output with theirs, so the final accuracy should be 100%
#Fits to my 4GB GPU, need ~1GB additionally for TSDF and ~500MB for OS
batchsize=20;
sumloss = 0;
numloss = 0;

#There are 654 test scenes
maxscenes = parse(Int32, ARGS[1]);
if maxscenes < 1
	maxscenes = 1;
elseif maxscenes > 654
	maxscenes = 654
end

#Choose VGGNet type, default to the paper's version
vggtype = parse(Int32, ARGS[2]);
if vggtype != 16 && vggtype != 19
	vggtype = 16;
end

#Choose model according to argument, free the other version's loaded weights
if vggtype == 16
	f2d=compile(:VGGNet16)
	#VGGWeights19 = 0;
elseif vggtype == 19
	f2d=compile(:VGGNet19)
	#VGGWeights16 = 0;
end
f3d=compile(:ORNFeature)
fClass=compile(:ORNClass)

#Entire network architecture
function Network(x2d, x3d)
	#Compute feature vectors
	v2d=forw(f2d, x2d)
	v3d=forw(f3d, x3d)

	#Resize from [1,1,4096,batchsize] to [4096,batchsize]
	v2d=reshape(v2d, size(v2d));

	#Predict class
	pClass=forw(fClass, v2d, v3d)

	return pClass;
end

println("Number of scenes to be processed is: $maxscenes\n");
println("***Start time: $(now())***");
@startTime("***Evaluating the test set...***\n");
for i=1:maxscenes
	@startTime("Preparing 3D Input Data...");
	#Compute TSDF using CUDA
	run(`./tsdf $i`);

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
		x2d=x2D[:,:,:, sx:ex];
		x3d=x3D[:,:,:,:, sx:ex];

		y_predict = Network(x2d, x3d);

		sumloss += zeroone(y_predict, hot_vec[:, sx:ex, i]);
		numloss += 1;

		loss = sumloss/numloss;

		#Print accuracy at the end of each batch
		println("Scene: $i Batch: $(convert(Int32,j)) Accuracy: $((1-loss)*100)%");
	end
	@stopTime("Calculation completed.")
end
@stopTime("***Test set evaluated.***");
println("***End time: $(now())***");
