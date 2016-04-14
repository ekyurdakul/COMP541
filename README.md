**Paper Implemantation**  
[Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images by Shuran Song and Jianxiong Xiao](http://dss.cs.princeton.edu/paper.pdf)  
[Used some C/C++/CUDA and MATLAB code of said paper's implemenation](https://github.com/shurans/DeepSlidingShape)  

**TESTED ON**  
Ubuntu 15.10  

**HOW TO RUN**  
julia Test.jl 2>&1 | tee test_output.txt  

**HOW TO INSTALL**  

*Read the README.md in the "data" folder to download and setup the data*  

*REQUIREMENTS*:  

[Knet](http://knet.readthedocs.org/en/dev/install.html)  
[Julia Package "MAT"](https://github.com/simonster/MAT.jl)   
7zip (For extracting processed data)  
Octave (Free alternative to MATLAB for preparing images)   
Octave package "image"   


*OCTAVE INSTALLATION*  

Execute the following commands in the terminal:  
*sudo apt-get install octave*  
*sudo apt-get install octave-image*  
*sudo apt-get install liboctave-dev*  

Launch Octave  
*sudo octave-cli*  

Execute the following in the Octave command line:  
*pkg install -forge image*  

**TEST SET**  
I chose NYU as my test set since it has less scenes compared to SUNRGBD (654 vs 5050), and each scene has a maximum of 2000 bounding boxes in it and this way it takes less time to get results.  
*SUNRGBD contains NYU as well*  

**DIFFERENCES**  
The paper uses the 16 layer version of VGGNet, mine has 19 layers.  
The paper makes use of 7x7 Region-of-Interest (RoI) pooling, however I crop/resize inputs to 224x224 since Knet does not have RoI pooling.  

*RoI Pooling*  
http://arxiv.org/pdf/1504.08083.pdf Fast R-CNN  
http://arxiv.org/pdf/1406.4729.pdf Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition  
http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf Fast R-CNN RoI Pooling Layer  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/include/caffe/fast_rcnn_layers.hpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cu  
