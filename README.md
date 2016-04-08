**HOW TO INSTALL**  

*Requirements*:  
*Knet  
*HDF5  
*7zip  

*Prepare the data*:  
1.Unzip all 7zip files in the "data" folder  


*RoI Pooling Layer*
http://arxiv.org/pdf/1504.08083.pdf  
http://mp7.watson.ibm.com/ICCV2015/slides/iccv15_tutorial_training_rbg.pdf  
http://arxiv.org/pdf/1406.4729.pdf  
https://pdfs.semanticscholar.org/8f67/64a59f0d17081f2a2a9d06f4ed1cdea1a0ad.pdf  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/include/caffe/fast_rcnn_layers.hpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cu  
roi_pooling_param {  
    pooled_w: 7  
    pooled_h: 7  
    spatial_scale: 0.0625  
  }  

http://arxiv.org/pdf/1409.1556.pdf VGGNet  
http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf FAST RCNN RoI Pooling Layer  

**TODO**  

Convert data into Julia compatible format  
*Test data  
*Trained weights  

**QUESTIONS**  

**Training options**  
L2 regularizer  
Momentum  
Weight decay  
Reduce learning rate every 5k iterations  

**Training**  
17 hours on K40 GPU  
"half" used instead of float, Knet uses double?  
4.27GB memory usage WITHOUT VGGnet  
