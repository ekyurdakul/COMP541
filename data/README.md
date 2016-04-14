**DATA SETUP**  

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
