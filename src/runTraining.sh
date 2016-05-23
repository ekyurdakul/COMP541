#julia Train.jl 2 16 10 2>&1 | tee ../experiments/vgg_16_layers_train_02_scenes_10_epochs_output.txt
#julia Train.jl 2 19 10 2>&1 | tee ../experiments/vgg_19_layers_train_02_scenes_10_epochs_output.txt


#julia Train.jl 10 16 10 2>&1 | tee ../experiments/vgg_16_layers_train_10_scenes_10_epochs_output.txt
#julia Train.jl 10 19 10 2>&1 | tee ../experiments/vgg_19_layers_train_10_scenes_10_epochs_output.txt


julia Train.jl 795 16 10 2>&1 | tee ../experiments/vgg_16_layers_train_all_scenes_10_epochs_output.txt
julia Train.jl 795 19 10 2>&1 | tee ../experiments/vgg_19_layers_train_all_scenes_10_epochs_output.txt
