all:
	nvcc -std=c++11 -o tsdf tsdf.cu
	nvcc -std=c++11 -o tsdf_train tsdf_train.cu

clean:
	-rm tsdf tsdf_train