#There are 3 arguments:  
#"sceneCount" limits the number of scenes to be processed  
#"VGGNetLayerCount" determines the type of VGGNet; either 16 or 19 layer version  
#"file" contains the output of the terminal
#Example:
#julia Test.jl "sceneCount" "VGGNetLayerCount" 2>&1 | tee "file"


#EXPERIMENTS

#2 scenes
#julia Test.jl 2 16 2>&1 | tee ../experiments/vgg_16_layers_test_02_scenes_output.txt
#julia Test.jl 2 19 2>&1 | tee ../experiments/vgg_19_layers_test_02_scenes_output.txt


#10 scenes
#julia Test.jl 10 16 2>&1 | tee ../experiments/vgg_16_layers_test_10_scenes_output.txt
#julia Test.jl 10 19 2>&1 | tee ../experiments/vgg_19_layers_test_10_scenes_output.txt


#All scenes
#julia Test.jl 654 16 2>&1 | tee ../experiments/vgg_16_layers_test_all_scenes_output.txt
#julia Test.jl 654 19 2>&1 | tee ../experiments/vgg_19_layers_test_all_scenes_output.txt
