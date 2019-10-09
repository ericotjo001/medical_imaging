# Self-Reflection GAN

**[Very Important]** Poor performance. In development. 

**Motivation**. Medical images are sometimes hard to obtain. For example, in 
<a href="http://isles-challenge.org/ISLES2017/">ISLES2017</a>, there are only 43 training cases. 
To overcome this problem, medical images can be synthetically generated for example by using 
<a href="https://arxiv.org/abs/1406.2661">GAN</a>. However, synthetic medical images might be hard
to interpret, especially since it might not sufficiently reflect real organs. Thus, we develop
solutions that use help to augment network learning in adversarial manner without generating synthetic images.

**Data**. CIFAR10. https://www.cs.toronto.edu/~kriz/cifar.html

**Usage**. To read the usage instruction, simply run the following from the working directory.
'''
python main.py
'''

**Initial design**. The following shows the initial development plan. *The code may no longer reflect the initial design* after several tests are conducted.[Update 20190910] Tested Gram-Schmidt-like procedure on the convolution filters. 

<div align="center">
<img  width="400" height="250" src="https://github.com/etjoa003/medical_imaging/blob/master/srgan/images/initial_design.JPG?raw=true">
  </div>
