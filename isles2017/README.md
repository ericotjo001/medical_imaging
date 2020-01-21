# Ischemic Stroke Lesion Segmentation
**Description**. This folder contains implementation of neural networks for ischemic stroke lesion segmentation from ISLES 2017. 
<br>

**Current status:** On pre-print.<br>
**Title:** Enhancing the Extraction of Interpretable Information for Ischemic Stroke Imaging from Deep Neural Networks.<br>
**Pre-print link:** https://arxiv.org/abs/1911.08136 .<br>

**Usage**. To find out the usage instruction, run the following from the working directory:
```
python main.py
```
<br>

**Objective**. The ultimate aim of this project is to provide a pipeline for diagnosis/prognosis using neural network. Furthermore, in healthcare, the interpretability of the neural network is important. In another words, we want to know **why** neural network predicts the output it predicts. See my <a href="https://arxiv.org/abs/1907.07374">survey paper</a> on the importance of interpretability of machine learning algorithms and neural networks.
<br>

**Data**. ISLES 2017 multi-modal MRI scans, 43 training cases, 32 test cases. Visit http://www.isles-challenge.org/ISLES2017/ for details.
<br>

**LRP (Layer-wise Relevance Propagation)**. LRP is a visualization tool developed to observe the importance of segments or parts of the input to the prediction made by a neural network. Visit http://heatmapping.org for more details. Notice that there are several variants of LRP applicable to different layers. We also adapted our algorithm from sources such as 
1. the older version of the main LRP site http://web.archive.org/web/20190529054742/http://heatmapping.org/tutorial/
2. and https://github.com/Hey1Li/Salient-Relevance-Propagation.
<br>

Some relevant figures:
<br>

<div align="center">
  <img width="640" height="250" src="https://github.com/etjoa003/medical_imaging/blob/master/isles2017/_others/for_show_scans.jpg?raw=true">
</div>

*Figure 1*. (left) ADC image of a patient from one of the training case. (right) Ground truth segmentation output. This shows the location of the lesion.

<div align="center">
  <img width="500" height="500" src="https://github.com/etjoa003/medical_imaging/blob/master/isles2017/_others/LatestImages/lrpfilter.JPG?raw=true">
</div>

*Figure 2*. (A) Raw LRP output. (B,C) LRP output with filters applied. (A2-C2) LRP output overlaid on a slice of RBF modality of an MRI scan. (D) Ground-truth lesion segmentation (E) Predicted lesion (F) A slice of RBF modality of an MRI scan.

<br>

Tips and notes:
+ utils/utils.py header contains all the imported packages used in this project, hence all the dependencies could be found there.
+ All tests are conducted in Windows 10, python 3.6. 

Warning (!)
1.
The older version of the main LRP site (see link above) has been replaced along with the older version of LRP formula. 
2.
Refer to models/networks_LRP.py. Line 128 has been updated to 
  tempn = np.minimum(0,self.weight.data.clone().cpu().detach().numpy())
from 
  tempn = np.maximum(0,self.weight.data.clone().cpu().detach().numpy())
The former is the suggested application of LRP on the input layer from the older version of the main LRP site (see link above). The published results here show the usage of the wrong version of the suggestion. The effect of the mistake is the amplification of positive weight at the final LRP layer and the coupling of signals associated to negative weights to the positive weights. This happens only at the input layer. We can see this as a variation of LRP application. We are currently investigating if similar "spiking errors" are observed in both the latest version of LRP (from the main website) and the "correct" version of older LRP algorithm. As of now, the main idea of "spiking errors" appear to still hold (to be updated), indicating that the error has little effects on the overall behaviour.
