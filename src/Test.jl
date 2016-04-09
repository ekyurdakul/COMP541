include("Network.jl")

@startTime("Loading input data...")

x2d=zeros(Float32, 224,224,3,1);
x3d=zeros(Float32, 30,30,30,3,1);

@stopTime("Loaded input data.")

@startTime("Sending input to the network...")
Network(x2d, x3d);
@stopTime("Calculation completed.")


#Call CUDA code
#y=ccall( (:calculateTSDF, "tsdf.so"), Int32, (Int32, Int32), 0, 5)
#y=ccall( (:calculateTSDF, "tsdf.so"), Void, (Int32, Int32), 0, 5)
