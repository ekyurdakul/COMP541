**Paper Implemantation**  
[Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images by Shuran Song and Jianxiong Xiao](http://dss.cs.princeton.edu/paper.pdf)  
[Used some C/C++/CUDA and MATLAB code of said paper's implemenation](https://github.com/shurans/DeepSlidingShape)  

**Tested on**  
Ubuntu 15.10  

**Results**  
VGGNet 19 Layers -> Scenes: 654 Accuracy: 89.89831804281107% Time: 39391.945318995 seconds  

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
julia Test.jl 2 16 2>&1 | tee ../experiments/vgg_16_layers_test_2_scenes_output.txt  
julia Test.jl 2 19 2>&1 | tee ../experiments/vgg_19_layers_test_2_scenes_output.txt  

julia Test.jl 654 16 2>&1 | tee ../experiments/vgg_16_layers_test_all_scenes_output.txt  
julia Test.jl 654 19 2>&1 | tee ../experiments/vgg_19_layers_test_all_scenes_output.txt  

**Test set**  
I chose NYU as my test set since it has less scenes compared to SUNRGBD (654 vs 5050), and each scene has a maximum of 2000 bounding boxes in it, so this way it takes less time to get results.  
*Note: NYU is a subset of SUNRGBD database*  

**Differences**  
The paper uses the 16 layer version of VGGNet, mine has also the 19 layer version.  
The paper makes use of 7x7 Region-of-Interest (RoI) pooling, however I crop/resize inputs to 224x224 since Knet does not have RoI pooling.  

*RoI Pooling*  
http://arxiv.org/pdf/1504.08083.pdf Fast R-CNN  
http://arxiv.org/pdf/1406.4729.pdf Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition  
http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf Fast R-CNN RoI Pooling Layer  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/include/caffe/fast_rcnn_layers.hpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cu  
