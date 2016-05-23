**Paper Implemantation**  
[Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images by Shuran Song and Jianxiong Xiao](http://dss.cs.princeton.edu/paper.pdf)  
[Used some C/C++/CUDA and MATLAB code of said paper's implemenation](https://github.com/shurans/DeepSlidingShape)  

**Tested on**  
Ubuntu 15.10  

**How to Install**  

[Knet](http://knet.readthedocs.org/en/dev/install.html)  
[Julia Package "MAT"](https://github.com/simonster/MAT.jl)   
[Julia Package "ImageMagick"](https://github.com/JuliaIO/ImageMagick.jl)  
7zip (For extracting processed data)  

*Read the README.md in the "data" folder to download and setup the data*  

**How to Run**  

Desired training experiments in /src/runTraining.sh can be uncommented and run by executing the file.  
The same is true for testing experiments. /src/runTesting.sh can be modified by uncommenting desired experiments and run.  

**Train & Test Set**  
I chose NYU as my train and test set since it has less scenes compared to SUNRGBD and this way it takes less time to get results.  
Number of scenes in the train set (NYU:795 vs SUNRGBD:5285 scenes)  
Number of scenes in the test set (NYU:654 vs SUNRGBD:5050 scenes)  

*Note: NYU is a subset of SUNRGBD database and each scene contains 2000 bounding boxes*  

**Results**  
Outputs of run experiments are located in /experiments/  

**Train Set Results**  
Batch size 	: 20  
Batches 	: 100  
Epochs		: 10  

Scenes : 2  
VGG 16 Layers -> Softloss: 2.9956963 Accuracy: 54.40000000000001% Time: 1220.596682183 seconds  
VGG 19 Layers -> Softloss: 2.995757 Accuracy: 53.35000000000003% Time: 1342.271687935 seconds  

Scenes : 10  
VGG 16 Layers -> Softloss: 2.9955368 Accuracy: 56.74000000000012% Time: 5783.906751494 seconds  
VGG 19 Layers -> Softloss: 2.995787 Accuracy: 55.01000000000005% Time: 6553.552235072 seconds  

*All results were obtained by the latest commit*  


**Test Set Results**  
Scenes 		: 654  
Batch size 	: 20  
Batches 	: 100  

VGG 16 Layers -> Accuracy: 89.87668195718415% Time: 36315.016382632 seconds  
VGG 19 Layers -> Accuracy: 89.89831804281107% Time: 39391.945318995 seconds  

*VGG 16 Layer results were obtained by commit afaf50ba8ab0c361b113ddf7363511c29783915b*  
*VGG 19 Layer results were obtained by commit cf6fb3ca63150382abc65373fd30c43d9ef974ca*  

**Differences**  
The paper uses the 16 layer version of VGGNet, mine also has the 19 layer version.  
The paper makes use of 7x7 Region-of-Interest (RoI) pooling, however I crop/resize inputs to 224x224 since Knet does not have RoI pooling.  


*RoI Pooling*  
http://arxiv.org/pdf/1504.08083.pdf Fast R-CNN  
http://arxiv.org/pdf/1406.4729.pdf Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition  
http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf Fast R-CNN RoI Pooling Layer  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/include/caffe/fast_rcnn_layers.hpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cu  
