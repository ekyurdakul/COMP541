include("Network.jl")

@startTime("Loading input data...")

x2d=zeros(Float32, 224,224,3,1);
x3d=zeros(Float32, 30,30,30,3,1);

@stopTime("Loaded input data.")

@startTime("Sending input to the network...")
Network(x2d, x3d);
@stopTime("Calculation completed.")
