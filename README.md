**TODO**

**Training options**
*L2 regularizer
*Momentum
*Weight decay
*Reduce learning rate every 5k iterations

**Layers**
*L1 Smooth

**Backprop**
*Multiple outputs? Not entirely feed forward
*Are gradients calculated correctly ?

**Training**
*17 hours on K40 GPU
*"half" used instead of float, Knet uses double?
*4.27GB memory usage WITHOUT VGGnet
*Reduce batchsize but increase epochs ???

**Data**
*Two dimensional batchsize [288, 96] ???
*Not clear if they trained on SUN or NYU or if SUN includes NYU as well
*data files are .list and .mat How to import into julia ?
*How to import VGGnet weights into julia? (.caffemodel)
*[3x30x30x30] input in weird .list files or not computed at all
