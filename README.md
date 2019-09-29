# medical_imaging
Collection of codes for medical imaging

## 1. isles2017

Current status: **[Update Ongoing]**

This folder contains implementation of neural networks for ischemic stroke lesion segmentation. 
<br>

To find out the usage instruction, run the following from the working directory:
```
python main.py
```
<br>

**Objective**. The ultimate aim of this project is to provide a pipeline for diagnosis/prognosis using neural network. Furthermore, in healthcare, the interpretability of the neural network is important. In another words, we want to know **why** neural network predicts the output it predicts. See my <a href="https://arxiv.org/abs/1907.07374">survey paper</a> on the importance of interpretability of machine learning algorithms and neural networks.
<br>

**Data**. ISLES 2017 multi-modal MRI scans, 43 training cases, 32 test cases. Visit http://www.isles-challenge.org/ISLES2017/ for details.
<br>

**LRP (Layer-wise Relevance Propagation)**. LRP is a visualization tool developed to observe the importance of segments or parts of the input to the prediction made by a neural network. Visit http://heatmapping.org for more details.
<br>

<div align="center">
  <img width="640" height="250" src="https://github.com/etjoa003/medical_imaging/blob/master/isles2017/_others/for_show_scans.jpg?raw=true">
</div>

*Figure 1*. (left) ADC image of a patient from one of the training case. (right) Ground truth segmentation output. This shows the location of the lesion.

<div align="center">
  <img width="400" height="250" src="https://github.com/etjoa003/medical_imaging/blob/master/isles2017/_others/LRP%20example%202.JPG?raw=true">
    <img width="400" height="250" src="https://github.com/etjoa003/medical_imaging/blob/master/isles2017/_others/LRP%20example.JPG?raw=true">
</div>

*Figure 2*. An ADC image (left) and a TTP image (right). LRP is used to find the importance of these images (amongst other channels) on the prediction output made by a model we train. Red/blue patch indicates positive/negative contribution. *Caveat*: at this point, the result is far from optimal.
<br>

Tips:
+ utils/utils.py header contains all the imported packages used in this project, hence all the dependencies could be found there.
<br>

Notes:
+ All tests are conducted in Windows 10, python 3.6. 

<div align="center">
  <img width="320" height="250" src="https://github.com/etjoa003/medical_imaging/blob/master/isles2017/past_results/20190827%20UNet3D/UNet3D_XXXXXX_loss_100.jpg?raw=true">
</div>


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
