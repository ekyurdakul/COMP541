**Paper Implemantation**  
[Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images by Shuran Song and Jianxiong Xiao](http://dss.cs.princeton.edu/paper.pdf)  
[Used some C/C++/CUDA and MATLAB code of said paper's implemenation](https://github.com/shurans/DeepSlidingShape)  

**Tested on**  
Ubuntu 15.10  

**Best Result**  
Scene: 10 Accuracy: 94.81500000000003% Time: 1563.439785184 seconds ~26 minutes  

**How to install**  

*Requirements*:  

[Knet](http://knet.readthedocs.org/en/dev/install.html)  
[Julia Package "MAT"](https://github.com/simonster/MAT.jl)   
[Julia Package "ImageMagick"](https://github.com/JuliaIO/ImageMagick.jl)  
7zip (For extracting processed data)  

*Read the README.md in the "data" folder to download and setup the data*  

**How to run**  
There are 2 arguments:  

"sceneCount" limits the number of scenes to be processed  
"file" contains the output of the terminal  

julia Test.jl "sceneCount" 2>&1 | tee "file"  

Example:  
julia Test.jl 2 2>&1 | tee ../experiments/test_2_scenes_output.txt  

**Test set**  
I chose NYU as my test set since it has less scenes compared to SUNRGBD (654 vs 5050), and each scene has a maximum of 2000 bounding boxes in it, so this way it takes less time to get results.  
*Note: NYU is a subset of SUNRGBD database*  

**Differences**  
The paper uses the 16 layer version of VGGNet, mine has 19 layers.  
The paper makes use of 7x7 Region-of-Interest (RoI) pooling, however I crop/resize inputs to 224x224 since Knet does not have RoI pooling.  

*RoI Pooling*  
http://arxiv.org/pdf/1504.08083.pdf Fast R-CNN  
http://arxiv.org/pdf/1406.4729.pdf Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition  
http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf Fast R-CNN RoI Pooling Layer  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/include/caffe/fast_rcnn_layers.hpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cu  
