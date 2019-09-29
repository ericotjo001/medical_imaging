# Ischemic Stroke Lesion Segmentation
**Description**. This folder contains implementation of neural networks for ischemic stroke lesion segmentation from ISLES 2017. 
<br>

Current status: **[Update Ongoing]**

**Usage**. To find out the usage instruction, run the following from the working directory:
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

Tips and notes:
+ utils/utils.py header contains all the imported packages used in this project, hence all the dependencies could be found there.
+ All tests are conducted in Windows 10, python 3.6. 
