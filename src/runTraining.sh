#There are 4 arguments:  
#"sceneCount" limits the number of scenes to be processed  
#"VGGNetLayerCount" determines the type of VGGNet; either 16 or 19 layer version  
#"epochs" number of epochs
#"file" contains the output of the terminal
#Example:
#julia Train.jl "sceneCount" "VGGNetLayerCount" "epochs" 2>&1 | tee "file"


#EXPERIMENTS

#2 scenes 10 epochs
#julia Train.jl 2 16 10 2>&1 | tee ../experiments/vgg_16_layers_train_02_scenes_10_epochs_output.txt
#julia Train.jl 2 19 10 2>&1 | tee ../experiments/vgg_19_layers_train_02_scenes_10_epochs_output.txt

#10 scenes 10 epochs
#julia Train.jl 10 16 10 2>&1 | tee ../experiments/vgg_16_layers_train_10_scenes_10_epochs_output.txt
#julia Train.jl 10 19 10 2>&1 | tee ../experiments/vgg_19_layers_train_10_scenes_10_epochs_output.txt


#All scenes 10 epochs
#julia Train.jl 795 16 10 2>&1 | tee ../experiments/vgg_16_layers_train_all_scenes_10_epochs_output.txt
#julia Train.jl 795 19 10 2>&1 | tee ../experiments/vgg_19_layers_train_all_scenes_10_epochs_output.txt
