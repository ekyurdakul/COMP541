julia Test.jl 2 16 2>&1 | tee ../experiments/vgg_16_layers_test_02_scenes_output.txt
julia Test.jl 2 19 2>&1 | tee ../experiments/vgg_19_layers_test_02_scenes_output.txt

julia Test.jl 10 16 2>&1 | tee ../experiments/vgg_16_layers_test_10_scenes_output.txt
julia Test.jl 10 19 2>&1 | tee ../experiments/vgg_19_layers_test_10_scenes_output.txt

julia Test.jl 654 16 2>&1 | tee ../experiments/vgg_16_layers_test_all_scenes_output.txt
julia Test.jl 654 19 2>&1 | tee ../experiments/vgg_19_layers_test_all_scenes_output.txt
