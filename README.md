# medical_imaging

2021 May Update: this repository has been restructured. Previous projects will be archived and no longer be updated.

Current collection of codes for medical imaging:
1. N.A. (No project for this repo is in progress)

Archived (no longer updated):
1. <a href="https://github.com/etjoa003/medical_imaging/tree/master/isles2017">isles2017</a>. Deep learning for ischemic stroke lesion segmentation.
2. <a href="https://github.com/etjoa003/medical_imaging/tree/master/multiv">Multiv</a>. Image slice sampler.
3. <a href="https://github.com/etjoa003/medical_imaging/tree/master/Attention-Gated-Networks_auxiliary">Attention-Gated-Networks_auxiliary</a>. Auxiliary codes for Attention Gated Networks.

<div align="center">
   <img width="300" height="200" src="https://github.com/etjoa003/medical_imaging/blob/master/images/lrpinput.jpg?raw=true">
   <img width="300" height="200" src="https://github.com/etjoa003/medical_imaging/blob/master/images/lrpoutput.JPG?raw=true">
  </div>
<br>

*Figure 1*. (left) An image input showing a castle. (right) The output of LRP to "explain" the decision of the algorithm to categorize the image into the correct class "castle". The images are obtained from the original website http://www.heatmapping.org/tutorial/.

<div align="center">
  <img width="500" height="500" src="https://github.com/etjoa003/medical_imaging/blob/master/isles2017/_others/LatestImages/lrpfilter.JPG?raw=true">
</div>

*Figure 2*. (A) Raw LRP output. (B,C) LRP output with filters applied. (A2-C2) LRP output overlaid on a slice of RBF modality of an MRI scan. (D) Ground-truth lesion segmentation (E) Predicted lesion (F) A slice of RBF modality of an MRI scan.

<br>

<div align="center">
  <img width="400" height="250" src="https://github.com/etjoa003/medical_imaging/blob/master/multiv/Image%20Store/dualview2D_test.gif?raw=true">
  </div>
<br>

*Figure 3*. Visualization of automated sampler.

For more details, visit each project folder.

<br>

Related links:
1. Layerwise Relevance Propagation. http://heatmapping.org
2. https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/utils/util.py
3. http://www.isles-challenge.org/ISLES2017/
