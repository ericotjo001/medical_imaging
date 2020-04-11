import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *
from utils.custom_augment import Image3DAugment

# DEMO = 'rotate'
# DEMO = 'translate'
DEMO = 'augment' # main demo

def demo_augment(x):
	aug = Image3DAugment(x.shape,verbose=250)
	aug_param = aug.generate_random_augment_params(verbose=250)
	x_mod = aug.rotate_then_clip_translate(x, aug_param)
	'''
	Recall that x, x_mod are both in w,h,d shapes
	Tranpose them first because the parent classes assume d,w,h shapes
	'''
	x = x.transpose(2,1,0)
	x_mod = x_mod.transpose(2,1,0)

	aug.vis2(x.transpose(1,2,0), x_mod.transpose(1,2,0))

def get3Dx(d, h, w):
	x = np.zeros(shape=(w,h,d))
	x = x.reshape((d,h,w))
	for i in range(d):
		x[i, int(h/2):int(h/1.8)+i, int(w/2):int(w/1.1)+i] = 1.8*d -i 
		x[i, int(h/2):int(h/1.2)+i, int(w/2):int(w/1.7)+i] = 1.8*d -i 
		x[i, int(h/1.4):int(h/1.3)+d, int(w/4):int(w/2)+i] = 1.8*d -i 
	x = np.abs(x/np.max(x))
	return x

def demo_rotate(x,theta_deg=15.,ax=0):
	vix = Visualizer3D()
	x_mod = vix.rotate_img3D(x, theta_deg, ax=ax)
	
	x_set = sorted(set(x.reshape(-1).tolist()))
	x_mod_set = sorted(set(x_mod.reshape(-1).tolist()))
	print(x_set)
	print(x_mod_set)
	if not x_set==x_mod_set: print("Unique set of values do change!")
	vix.vis2(x.transpose(1,2,0), x_mod.transpose(1,2,0))

def demo_translate(x, pixel=1, clip_min_max = [0,1], ax=0):
	ct = Visualizer3D()
	x_mod = ct.clip_and_translate(x, pixel=pixel, clip_min_max = clip_min_max, ax=ax)
	ct.vis2(x.transpose(1,2,0), x_mod.transpose(1,2,0))



if __name__=='__main__':
	w,h,d= 64,48,9
	# w,h,d = 4,4,2
	x = get3Dx(d,h,w)
	if DEMO == 'rotate': demo_rotate(x,theta_deg = 15.,ax=0)
	elif DEMO == 'translate': demo_translate(x, pixel=10, clip_min_max = [0,1.], ax=1)
	elif DEMO == 'augment': demo_augment(x.transpose(2,1,0))





	
