import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *
import utils.augment as au

class Image3DAugment(au.Visualizer3D, au.Image3DRotator, au.Image3DClipTranslate):
	'''
	Let this class of object be the custom class
	IMPT: ASSUME OBJECT x is in 3d shape w,h,d. NOTE that the parent classes
	  assume object x have d,h,w shape 
	In this custom funciton, rotate only w.r.t the depth axis
	'''
	def __init__(self, shape3D, verbose=0):
		super(Image3DAugment, self).__init__()
		self.verbose = verbose
		s = shape3D
		percent_translate = 0.2
		self.theta_range_depth_axis = [-20.,20.] # range to rotate w.r.t depth axis in degree
		self.pixel_range_depth_axis = [-np.ceil(s[2]*percent_translate), np.ceil(s[2]*percent_translate)]
		self.pixel_range_height_axis = [-np.ceil(s[1]*percent_translate), np.ceil(s[1]*percent_translate)]
		self.pixel_range_width_axis = [-np.ceil(s[0]*percent_translate), np.ceil(s[0]*percent_translate)]
		self.crop_range = [0., 1.]

		if self.verbose>249:
			print("Image3DAugment. shape:%s"%(str(s)))
			print("  theta_range_depth_axis:%s"%(str(self.theta_range_depth_axis)))
			print("  pixel_range_depth_axis:%s"%(str(self.pixel_range_depth_axis)))
			print("  pixel_range_height_axis:%s"%(str(self.pixel_range_height_axis)))
			print("  pixel_range_width_axis:%s"%(str(self.pixel_range_width_axis)))
			print("  crop_range:%s"%(str(self.crop_range)))

	def generate_random_augment_params(self, verbose=0):
		aug_param = {}
		aug_param['theta_depth_axis'] = round(np.random.uniform(self.theta_range_depth_axis[0],self.theta_range_depth_axis[1]),1)
		aug_param['pixel_depth_axis'] = int(np.floor(np.random.uniform(self.pixel_range_depth_axis[0],self.pixel_range_depth_axis[1])))
		aug_param['pixel_height_axis'] = int(np.floor(np.random.uniform(self.pixel_range_height_axis[0],self.pixel_range_height_axis[1])))
		aug_param['pixel_width_axis'] = int(np.floor(np.random.uniform(self.pixel_range_width_axis[0], self.pixel_range_width_axis[1])))
		aug_param['crop'] = [0, round(np.random.uniform(0.3, self.crop_range[1]),2)]

		if verbose>249:
			for xkey in aug_param: print("param[%s] : %s"%(str(xkey), str(aug_param[xkey])))
		return aug_param

	'''
	config_data['augmentation']['type'] 
	'''
	# (0)
	def no_augmentation(self,x):
		return x

	# (1)
	def rotate_then_clip_translate(self, x, aug_param):
		'''
		Recall: Assume x is in 3d shape w,h,d
		Recall: the inherited functions assume d,h,w instead
		Therefore: do transpose!
		Assume input is normalized to [0,1]
		'''
		x1 = x.transpose(2,1,0)
		x1 = self.rotate_img3D(x1, aug_param['theta_depth_axis'], ax=0) 
		x1 = self.clip_and_translate(x1, pixel=aug_param['pixel_depth_axis'], clip_min_max = [0,1], ax=0)
		x1 = self.clip_and_translate(x1, pixel=aug_param['pixel_height_axis'], clip_min_max = [0,1], ax=1)
		x1 = self.clip_and_translate(x1, pixel=aug_param['pixel_width_axis'], clip_min_max = aug_param['crop'], ax=2)
		return x1.transpose(2,1,0)


