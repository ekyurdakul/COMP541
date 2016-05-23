**Paper Implemantation**  
[Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images by Shuran Song and Jianxiong Xiao](http://dss.cs.princeton.edu/paper.pdf)  
[Used some C/C++/CUDA and MATLAB code of said paper's implemenation](https://github.com/shurans/DeepSlidingShape)  

**Tested on**  
Ubuntu 15.10  

**Test Set Results**  
VGG 16 Layers -> Scene: 654 Batch: 100 Accuracy: 89.87668195718415% Time: 36315.016382632 seconds  
VGG 19 Layers -> Scene: 654 Batch: 100 Accuracy: 89.89831804281107% Time: 39391.945318995 seconds  

VGG 16 Layer results were obtained by the latest commit  
VGG 19 Layer results were obtained by commit cf6fb3ca63150382abc65373fd30c43d9ef974ca  

**How to install**  

*Requirements*:  

[Knet](http://knet.readthedocs.org/en/dev/install.html)  
[Julia Package "MAT"](https://github.com/simonster/MAT.jl)   
[Julia Package "ImageMagick"](https://github.com/JuliaIO/ImageMagick.jl)  
7zip (For extracting processed data)  

*Read the README.md in the "data" folder to download and setup the data*  

**How to run**  
There are 3 arguments:  

"sceneCount" limits the number of scenes to be processed  
"VGGNetLayerCount" determines the type of VGGNet; either 16 or 19 layer version  
"file" contains the output of the terminal  

julia Test.jl "sceneCount" "VGGNetLayerCount" 2>&1 | tee "file"  

Examples:  
julia Test.jl 2 16 2>&1 | tee ../experiments/vgg_16_layers_test_02_scenes_output.txt  
julia Test.jl 2 19 2>&1 | tee ../experiments/vgg_19_layers_test_02_scenes_output.txt  

julia Test.jl 10 16 2>&1 | tee ../experiments/vgg_16_layers_test_10_scenes_output.txt  
julia Test.jl 10 19 2>&1 | tee ../experiments/vgg_19_layers_test_10_scenes_output.txt  

julia Test.jl 654 16 2>&1 | tee ../experiments/vgg_16_layers_test_all_scenes_output.txt  
julia Test.jl 654 19 2>&1 | tee ../experiments/vgg_19_layers_test_all_scenes_output.txt  

**Test set**  
I chose NYU as my test set since it has less scenes compared to SUNRGBD (654 vs 5050), and each scene has a maximum of 2000 bounding boxes in it, so this way it takes less time to get results.  
*Note: NYU is a subset of SUNRGBD database*  

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
