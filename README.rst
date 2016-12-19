COMP 541 - Machine Learning Term Project
========================================

Paper Implementation
####################

Implementation of the Object Recognition Network (ORN) in [1]_.

Index
-----
- `Installation`_
- `Data Setup`_
- `Run Training`_
- `Run Testing`_
- `Training Results (from scratch)`_
- `Testing Results (with pretrained weights)`_
- `Experiments`_
- `Differences`_
- `References`_

Installation
------------

#) Install `7zip <http://www.7-zip.org/download.html>`_
#) Install `Julia <http://julialang.org/downloads/>`_
#) Execute in terminal

::

    cd src/
    make all
    make packages

Data Setup
----------
**Warning:** Total size ~30 GB.

#) Download and extract `this file <http://rgbd.cs.princeton.edu/data/SUNRGBD.zip>`__ to **data/**
#) Download `this file <http://dss.cs.princeton.edu/Release/result/proposal/RPN_NYU/boxes_NYU_po_test_nb2000_fb.list>`__ to **data/**
#) Download `this file <http://dss.cs.princeton.edu/Release/result/proposal/RPN_NYU/boxes_NYU_po_train_diff_nb2000_fb.list>`__ to **data/**
#) Download all files under `here <http://dss.cs.princeton.edu/Release/sunrgbd_dss_data/>`__ to **data/**
#) Extract all .7z in **data/julia_data** and delete all .7z

NYUv2 dataset is used, which has 795 train and 654 test scenes.

Run Training
------------

Execute in terminal

::

    julia src/Train.jl scenes VGGModel epochs

where,

scenes
    number of scenes to be processed, in range of [1,795]
VGGModel
    16,19
epochs
    number of epochs in range of [1,∞)

Examples:

::

    julia src/Train.jl 2 16 10
    julia src/Train.jl 795 19 10


Run Testing
-----------

Execute in terminal

::

    julia src/Test.jl scenes VGGModel

where,

scenes
    number of scenes to be processed, in range of [1,654]
VGGModel
    16,19


Examples:

::

    julia src/Test.jl 2 16
    julia src/Test.jl 654 19

Training Results (from scratch)
-------------------------------

+-----+----------+----------+------------+
| VGG | Softloss | Accuracy | Time       |
+=====+==========+==========+============+
| Scenes = 2 Batchsize = 20 Epochs = 10  |
+-----+----------+----------+------------+
| 16  | 2.9956963| 54.40%   | 00:20:21   |
+-----+----------+----------+------------+
| 19  | 2.995757 | 53.35%   | 00:22:23   |
+-----+----------+----------+------------+
| Scenes = 10 Batchsize = 20 Epochs = 10 |
+-----+----------+----------+------------+
| 16  | 2.9955368| 56.74%   | 01:36:24   |
+-----+----------+----------+------------+
| 19  | 2.995787 | 55.01%   | 01:49:14   |
+-----+----------+----------+------------+

Testing Results (with pretrained weights)
-----------------------------------------

+-----+----------+------------+
| VGG | Accuracy | Time       |
+=====+==========+============+
| Scenes = 654 Batchsize = 20 |
+-----+----------+------------+
| 16  | 89.8767% | 10:05:15   |
+-----+----------+------------+
| 19  | 89.8983% | 10:56:32   |
+-----+----------+------------+

Experiments
-----------
Outputs of experiments are in **experiments/**

Differences
-----------
- Paper uses only the 16 layer VGGNet.
- Paper uses 7x7 Region of Interest (RoI) pooling; I resize inputs, since Knet does not support RoI pooling.

References
----------
.. [1] \S. Song, J. Xiao. Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images. https://arxiv.org/abs/1511.02300. https://github.com/shurans/DeepSlidingShape