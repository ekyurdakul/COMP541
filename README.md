**Paper Implemantation**  
*Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images* by Shuran Song and Jianxiong Xiao  
*Link* http://dss.cs.princeton.edu/paper.pdf  

Used some C/C++/CUDA and MATLAB code of said paper's implemenation  
*Link* https://github.com/shurans/DeepSlidingShape  

**DONE**  

Accuracy implemented  

**TODO**  
Scene iteration "i" incorrect, should be "sceneID"  
Recheck Trained weights  

**HOW TO INSTALL**  
*Requirements*:  
Julia Package "MAT" @ https://github.com/simonster/MAT.jl  
7zip (For extracting processed data)  
Octave (Free alternative to MATLAB)   
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


*Tested on*  
Ubuntu 15.10  

*Warning*  
Total size of the data is around ~25 GB  

*Prepare the data*:  
1)Download and extract this project  
2)Download and extract: http://rgbd.cs.princeton.edu/data/SUNRGBD.zip  
3)Download and extract: http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip  
4)Download and extract: https://github.com/shurans/DeepSlidingShape/archive/master.zip  
5)Arrange the extracted folders such that the project directory looks like the following  
"Comp541-Term-Project-master/data/julia_data"  
"Comp541-Term-Project-master/data/marvin/"  
"Comp541-Term-Project-master/data/matlab_code"  
"Comp541-Term-Project-master/data/SUNRGBD"  
"Comp541-Term-Project-master/data/SUNRGBDtoolbox"  
6)Move "Comp541-Term-Project-master/octave/downloadData.m" to "Comp541-Term-Project-master/data/matlab_code" and overwrite it  
7)Launch MATLAB/Octave  
8)Change working directory to "Comp541-Term-Project-master/data/matlab_code"  
9)Execute the following two commands  
downloadData('Comp541-Term-Project-master/data','dss.cs.princeton.edu/Release/sunrgbd_dss_data/','.bin');  
downloadData('Comp541-Term-Project-master/data','dss.cs.princeton.edu/Release/image/','.tensor');  


**HOW TO RUN**  
julia Test.jl 2>&1 | tee test_output.txt  

**TEST SET**  
Each file has maximum of 2000 bounding boxes in it  
SUNRGBD 5050 files  
NYU	654  files  

*SUNRGBD contains NYU as well, therefore I chose NYU since it takes much less time to get results*  


**DIFFERENCES**  
The paper uses the 16 layer version of VGGNet, mine has 19 layers.  
The paper makes use of 7x7 Region-of-Interest (RoI) pooling, however I crop/resize inputs to 224x224 since Knet does not have RoI pooling.  

*RoI Pooling*  
http://arxiv.org/pdf/1504.08083.pdf  
http://mp7.watson.ibm.com/ICCV2015/slides/iccv15_tutorial_training_rbg.pdf  
http://arxiv.org/pdf/1406.4729.pdf  
https://pdfs.semanticscholar.org/8f67/64a59f0d17081f2a2a9d06f4ed1cdea1a0ad.pdf  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/include/caffe/fast_rcnn_layers.hpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cpp  
https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cu  
http://arxiv.org/pdf/1409.1556.pdf VGGNet  
http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf FAST RCNN RoI Pooling Layer  
