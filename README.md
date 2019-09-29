# medical_imaging
Collection of codes for medical imaging

## 1. isles2017

Current status: **[Update Ongoing]**

This folder contains implementation of neural networks for ischemic stroke lesion segmentation. The following site is where the challenge is described http://www.isles-challenge.org/ISLES2017/.

<div align="center">
  <img width="360" height="250" src="https://github.com/etjoa003/medical_imaging/blob/master/isles2017/past_results/20190827%20UNet3D/UNet3D_XXXXXX_loss_100.jpg?raw=true">
</div>

To find out the usage instruction, run the following from the working directory:
```
python main.py
```

Tips:
+ utils/utils.py header contains all the imported packages used in this project, hence all the dependencies could be found there.

Notes:
All tests are conducted in Windows 10, python 3.6. 


<div align="center">
  <img width="360" height="250" src="https://github.com/etjoa003/medical_imaging/blob/master/isles2017/past_results/20190827%20UNet3D/UNet3D_XXXXXX_loss_100.jpg?raw=true">
</div>
Evaluation is now performed only on training data. State-of-the-art algorithms still have difficulty attaining high dice score even on the training data. The following shows cross entropy loss for 100 epochs training on 43 patients without data augmentation. Dice score achieves 0.3+ (see past_results/20190827 UNet3D), and cross entropy loss above suggests further training can improve performance. Email me at ericotjoa@gmail.com if you would like the pretrained model (it is still far from optimal, though).

## 2. Multiv
While working on medical images, for example in NIFTI formats, we might face memory problem. This is because a NIFTI volume might come in large sizes, for example 192x192x19 with 6 modalities. With large convolutional neural network, feeding the entire volume may result in out of memory error (at least my measly 4GB RAM does. Multi-view sampling is the way out of this. Using multi-view sampling, slices of the images (green rectangles) are sampled. The "multi" part of the multi-view can take the form of larger slice (red rectangles).

This is inspired by https://arxiv.org/abs/1603.05959.

<div align="center">
  <img src="https://github.com/etjoa003/medical_imaging/blob/master/multiv/Image%20Store/dualview2D_test.gif">
</div>

### 2.1. Usage
Run the following in the comand line to run the tests: 
```
python multiv.py
```

See the implementations of the sampler class objects from multivtests.py or from the jupyter notebook tutorials.

Dependencies: read the imports in multivutils.py
Note that tests are conducted in Windows 10, python 3.6. 


## 3. Attention-Gated-Networks_auxiliary
Working on Attention-Gated-Networks from https://github.com/ozan-oktay/Attention-Gated-Networks,
you might want to run the algorithm on the original datasets. One of them is PANCREAS CT-82 which
can be found from https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT.

The .dcm files store 2D slices. This code combines (and normalize) the slices of each patient 
into 3D volume. Each patient is coded as PANCREAS_0001, PANCREAS_0002 etc.

### 3.1. Usage
You can run the following in the command line:   
```
python ctpan.py -h
```

Dependencies: read the utils.py or Attention-Gated-Networks_auxiliary/README.txt

Note that tests are conducted in Windows 10, python 3.6. 
