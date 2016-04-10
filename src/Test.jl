include("Network.jl")

#batchsize=1 uses ~2GB GPU Memory
batchsize=1;
#number of scenes to process
maxscenes=5;
#Call CUDA code for fast computation, initialize C variables first
ccall((:initTSDF,"tsdf.so"), Void, (Int32, Int32), 0, maxscenes);

for i=1:maxscenes
	#Compute TSDF using CUDA
	boxcount=Ref{Int32}(0);
	status=ccall((:getNextTSDF,"tsdf.so"), Int32, (Ref{Int32},), boxcount);
	boxcount=boxcount[];
	
	@startTime("Loading input data...")
	#2D Input
	x2D=zeros(Float32,224,224,3,boxcount);
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
		print("Batch $(convert(Int32,j)): ");
		x2d=x2D[:,:,:, sx:ex];
		x3d=x3D[:,:,:,:, sx:ex];	
		Network(x2d, x3d);
	end
	@stopTime("Calculation completed.")
end

#Free C memory
ccall((:freeTSDF,"tsdf.so"), Void, ());
