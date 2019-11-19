from utils.debug_switches import *

import os, argparse, sys, json, time, pickle, time, csv, collections, copy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from PIL import Image
from multiprocessing import Pool
from os import listdir
from contextlib import redirect_stdout

import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader



this_device = torch.device('cuda:0')
number_of_classes = 2 # 0,1

DESCRIPTION = '''\t\t\t=== Welcome to meim3/main.py! ===

Implementations of 3D versions of Neural Network to handle ISLES 2017 Ischemic Stroke Lesion Segmentation.
See http://www.isles-challenge.org/ISLES2017/
Send me email at ericotjoa@gmail.com if you would like to a sample of trained model.

Currently available:
1. U-Net + LRP. Baseline performance on training dataset can reach the state of the art. Dice score 0.3~0.6 

Minimalistic step-by-step instructions:
1. run python main.py --mode create_config_file --config_dir config.json 
2. edit the directories of your ISLES2017 data in the config.json
3. choose training_mode and evaluation_mode (see below)
4. run python main.py --mode train --config_dir config.json. The outputs should be in "checkpoints" folder.
5. run python main.py --mode evaluation --config_dir config.json. The outputs should be in "checkpoints" folder.

Tips: 
(+) See entry.py shortcut_sequence to create custom training sequences.
(+) DEBUG modes in utils/debug_switches.py are convenient for debugging. Try them out.

Layerwise Relevance Propagation (LRP). See this website: http://www.heatmapping.org/

Configurations:
  ** Warning: Not all entries in the configuration files are relevant to specific implementations. 
    For example, in one particular instance of our implementation, when learning mechanism is adam, 
    momentum is not used. When SGD is used as the learning mechanism, betas are not relevant.

The following can be found in the configuration file.
- training_mode. See train() [entry.py].

- data_submode. Generally, find this configuration in [training.py]

- data_modalities. Be aware that the loader is sensitive to the order of the modalities.
  Some loaders do have some assumptions (for example OT is included), do check it out.
  Mostly, consider the scripts in dataio/ folder.

- evaluation. See evaluation() [entry.py].

- debug_test_mode. See test() [tests/test.py]

- augmentation. See custom_augment.py, "section config_data['augmentation']['type']" .

Modes:
(1) info
  python main.py
  python main.py info

(2) create_config_file
  python main.py --mode create_config_file
  python main.py --mode create_config_file --config_dir config.json 

(3) test. Ad hoc testing.
  python main.py --mode test
  python main.py --mode test --config_dir config.json 

(4) train. Training network.
  python main.py --mode train
  python main.py --mode train --config_dir config.json

(5) evaluation. Use trained network to perfrom task.
  python main.py --mode evaluation
  python main.py --mode evaluation --config_dir config.json

(6) lrp. Use Layerwise Relevance Propagation (LRP) for interpretability studies. See description above.
  python main.py --mode lrp
  python main.py --mode lrp --config_dir config.json

(7) visual.
  python main.py --mode visual
  python main.py --mode visual --config_dir config.json

(X) shorcut_sequence
  python main.py --mode shortcut_sequence
  python main.py --mode shortcut_sequence --config_dir config.json
  python main.py --mode shortcut_sequence --config_dir config.json --shortcut_mode XX1
'''

CONFIG_FILE = {
	'working_dir':"D:/Desktop@D/meim2venv/meim3",
	'relative_checkpoint_dir':'checkpoints',
	'data_directory':{
		'dir_ISLES2017':"D:/Desktop@D/meim2venv/meim3/data/isles2017",
	},
	'data_submode':'load_many_cases_type0003',
	'data_modalities':['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT'],
	'model_label_name': 'UNet3D_AYXXX2',
	'training_mode': 'UNet3D',
	'evaluation_mode': 'UNet3D_overfit',
	'visual_mode':'lrp_UNet3D_overfit_visualizer',
	'lrp_mode': 'lrp_UNet3D_overfit',
	'debug_test_mode':"test_load_many_ISLES2017",
	
	'basic':{
		'batch_size' : 1,
		'n_epoch' : 2,
		'save_model_every_N_epoch': 1,
		'keep_at_most_n_latest_models':4
	},
	'dataloader':{
		'resize' : [48,48,19] # [192,192,19]
	},
	'learning':{
		'mechanism':"adam",
		'momentum':0.9,
		'learning_rate':0.0002,
		'weight_decay':0.00001,
		'betas':[0.5,0.9],
		},
	'normalization': {
		'ADC_source_min_max' : [None, None], # "[0, 5000]",
		'MTT_source_min_max' : [None, None],
		'rCBF_source_min_max' : [None, None],
		'rCBV_source_min_max' : [None, None],
		'Tmax_source_min_max' : [None, None],
		'TTP_source_min_max' : [None, None],
		
		'ADC_target_min_max' : [0,1],
		'MTT_target_min_max' : [0,1],
		'rCBF_target_min_max' : [0,1],
		'rCBV_target_min_max' : [0,1],
		'Tmax_target_min_max' : [0,1],
		'TTP_target_min_max' : [0,1],
	},
	'augmentation': {
		'type': 'no_augmentation',
		'number_of_data_augmented_per_case': 10,
	},
	'LRP':{
		'relprop_config':{
			'mode': 'UNet3D_standard',
			'normalization': 'raw',
			'UNet3D': {
				'concat_factors': [0.5,0.5,0.5]
			},
			'fraction_pass_filter' :{
				"positive":[0.0,0.6],
				"negative":[-0.6,-0.0]		
			},
			'fraction_clamp_filter' :{
				"positive":[0.0,0.6],
				"negative":[-0.6,-0.0]		
			},
		},
		'filter_sweeper' :{
			'submode' : '0001',
			'case_numbers': [4,27]
		}
	},
	'misc': {
		'case_number':1
	}
}

class ConfigManager(object):

	def __init__(self):
		super(ConfigManager, self).__init__()
		
	def create_config_file(self,config_dir):
		print("utils/utils.py. create_config_file()")

		temp_dir = "temp.json"
		with open(temp_dir, 'w') as json_file:  
			json.dump(CONFIG_FILE, json_file, separators=(',',":"), indent=2) # (',', ':')

		self.json_beautify(temp_dir, config_dir)
		print("  Config file created.")

	def json_file_to_pyobj(self,filename):
	    def _json_object_hook(d): return collections.namedtuple('ConfigX', d.keys())(*d.values())
	    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
	    return json2obj(open(filename).read())

	def json_file_to_pyobj_recursive_print(self,config_data, name='config_data', verbose=0):
		for field_name in config_data._fields:
			temp = getattr(config_data,field_name)
			if 'ConfigX' in str(type(temp)) : self.json_file_to_pyobj_recursive_print(temp, name+'.'+field_name)
			else: 
				print("  %s.%s: %s [%s]"%(name,field_name, str(temp), str(type(temp))))
				if isinstance(temp,list) and verbose>9: 
					for x in temp: print("      [%s]"%(type(x)),end='')
					print()
				
	def json_beautify(self,temp_dir, config_dir):
		bracket = 0	
		with open(config_dir, 'w') as json_file: 
			for x in open(temp_dir, 'r'):
				temp = x.strip("\n").strip("\t")
				prev_bracket = bracket
				if "]" in temp: bracket-=1 
				if "[" in temp: bracket+=1
				if bracket ==0 and not prev_bracket==1: print(temp); json_file.write(temp+"\n")
				elif bracket ==0 and prev_bracket==1: print(temp.strip(" ")); json_file.write(temp.strip(" ")+"\n")
				elif bracket==1 and prev_bracket==0: print(temp,end=''); json_file.write(temp)
				else: print(temp.strip(" "),end='') ; json_file.write(temp.strip(" "))
		os.remove(temp_dir)

	def recursive_namedtuple_to_dict(self, config_data):
		if not type(config_data) == type({}):
			config_data = config_data._asdict()
		for xkey in config_data:
			if 'ConfigX' in str(type(config_data[xkey])):
				config_data[xkey] = config_data[xkey]._asdict()
				for ykey in config_data[xkey]:
					if 'ConfigX' in str(type(config_data[xkey][ykey])):
						config_data[xkey][ykey] = self.recursive_namedtuple_to_dict(config_data[xkey][ykey])
		return config_data

class DictionaryTxtManager(object):
	def __init__(self, diary_dir, diary_name, dictionary_data=None, header=None):
		super(DictionaryTxtManager, self).__init__()
		self.diary_full_path = os.path.join(diary_dir,diary_name)
		if not os.path.exists(diary_dir): os.mkdir(diary_dir)
		
		self.diary_mode = 'a' 
		if not os.path.exists(self.diary_full_path): self.diary_mode = 'w'

		if dictionary_data is not None:
			self.write_dictionary_to_txt(dictionary_data, header=header)			

	def write_dictionary_to_txt(self, dictionary_data,header=None):
		txt = open(self.diary_full_path,self.diary_mode)
		if header is not None: txt.write(header)
		self.recursive_txt(txt,dictionary_data,space='')
		txt.close()

	def recursive_txt(self,txt,dictionary_data,space=''):
		for x in dictionary_data:
			if isinstance(dictionary_data[x],dict) or type(dictionary_data[x])==type(collections.OrderedDict({})):
				txt.write("%s%s :\n"%(space,x))
				# self.write_dictionary_to_txt(dictionary_data[x],space=space+'  ')
				# for y in dictionary_data[x]:
				# 	txt.write("%s%s :"%(space,y))
				self.recursive_txt(txt,dictionary_data[x], space=space+'  ')
					# txt.write("\n")
			else:
				txt.write("%s%s : %s [%s]"%(space,x, dictionary_data[x],type(dictionary_data[x])))
			txt.write("\n")

class Logger():
	"""
	Usage, run your program after running the following
	sys.stdout = Logger(full_path_log_file="hello.txt")
	"""
	def __init__(self, full_path_log_file="logfile.log"):
		self.terminal = sys.stdout
		self.log = open(full_path_log_file, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self): pass   

def read_csv(filename, header_skip=1,get_the_first_N_rows = 0):
    # filename : string, name of csv file without extension
    # headerskip : nonnegative integer, skip the first few rows
    # get_the_first_N_rows : integer, the first few rows after header_skip to be read
    #   if all rows are to be read, set to zero

    # example
    # xx = read_csv('tryy', 3)
    # xx = [[np.float(x) for x in y] for y in xx] # nice here
    # print(xx)
    # print(np.sum(xx[0]))
    
    out = []
    with open(str(filename)+'.csv', encoding='cp932', errors='ignore') as csv_file:
        data_reader = csv.reader(csv_file)
        count = 0
        i = 0
        for row in data_reader:
            if count < header_skip:
                count = count + 1
            else:
                out.append(row)
                # print(row)
                if get_the_first_N_rows>0:
                	i = i + 1
                	if i == get_the_first_N_rows:
                		break
    return out

def normalize_numpy_array(x,target_min=-1,target_max=1, source_min=None, source_max=None, verbose = 250):
	'''
	If target_min or target_max is set to None, then no normalization is performed
	'''
	if source_min is None: source_min = np.min(x)
	if source_max is None: source_max = np.max(x)
	if target_min is None or target_max is None: return x
	if source_min==source_max:
		if verbose> 249 : print("normalize_numpy_array: constant array, return unmodified input")
		return x
	midx0=0.5*(source_min+source_max)
	midx=0.5*(target_min+target_max)
	y=x-midx0
	y=y*(target_max - target_min )/( source_max - source_min)
	return y+midx


 
def interp3d(x,size, mode='nearest'):
	'''
	x is a 3D tensor

	# x = np.random.randint(0,10,size=(2,3,4))
	# x = torch.Tensor(x)
	# z = F.interpolate(x, size=(8), mode='nearest')
	# print(x.shape)
	# print(z)
	# print(interp3d(x,(2,3,4)))
	'''
	x = F.interpolate(x, size=(int(size[2])), mode=mode)
	x = x.transpose_(1,2)
	x = F.interpolate(x, size=(int(size[1])), mode=mode)
	x = x.transpose_(1,2)
	x = x.transpose_(0,2)
	x = F.interpolate(x, size=(int(size[0])), mode=mode)
	x = x.transpose_(0,2)
	return x

def batch_interp3d(x,size,mode='nearest'):
	'''
	x is of the form 5D, (batch_size,C,D,H,W)
	'''
	s = tuple(x.shape)
	out = torch.zeros((s[0],s[1])+ size)

	for i in range(s[0]):
		for j in range(s[1]):
			out[i,j,:,:,:] = interp3d(x[i,j,:,:,:],size, mode=mode)
	return out

def batch_no_channel_interp3d(x,size,mode='nearest'):
	'''
	x is of the form 4D, (batch_size,D,H,W)
	'''
	s = tuple(x.shape)
	out = torch.zeros((s[0],)+ size)

	for i in range(s[0]):
		out[i,:,:,:] = interp3d(x[i,:,:,:],size, mode=mode)
	return out

def get_zero_container(x,y):
	"""
	Assume x, y are tensors of the same dimension
	but not necessarily have the same size
	for example x can be 3,4,5 and y 4,3,5
	return the size that can contain both: 4,4,5
	
	Example:
	x = torch.tensor(np.random.normal(0,1,size=(3,4,5)))
	y = torch.tensor(np.random.normal(0,1,size=(4,3,5)))
	z = get_zero_container(x,y)
	print(z.shape) # torch.Size([4, 4, 5])
	"""
	s = []
	for sx,sy in zip(x.shape,y.shape):
		s.append(np.max([sx,sy]))
	return torch.zeros(s)


def centre_crop_tensor(x, intended_shape):
	"""
	Assume x is pytorch tensor
	Given tensor x of shape (4,6), if we want to extract
	  its center of size (4,4), then set intended_shape=(4,4)
	Any dimension of x works.
	Example:
		x = torch.tensor(np.random.randint(0,10,size=(4,6)))
		x1 = centre_crop_tensor(x, (4,4))
		print(x)
		print(x1)
	Example 2:
	for i in range(1000):
		randomsize = np.random.randint(50,100,size=(1,2))[0]
		randomcropsize = np.random.randint(10,48,size=(1,2))[0]
		x = torch.tensor(np.random.randint(0,10,size=randomsize))
		x1 = centre_crop_tensor(x, randomcropsize)
		print("%s|| %s == %s"%(str(randomsize),str(randomcropsize),str(np.array(x1.shape))))

	"""
	limits = []
	for s,s1 in zip(x.shape,intended_shape):
		if s == s1: 
			limits.append(slice(0,s,None))
		else:
			diff1 = np.floor(np.abs(s-s1)/2)
			diff2 = np.ceil(np.abs(s-s1)/2)
			limits.append(slice(int(diff1),int(s-diff2),None))
	limits = tuple(limits)
	x1 = x[limits]
	# print(np.array(x1.shape),intended_shape)
	assert(np.all(np.array(x1.shape)==intended_shape))
	return x1



class SaveableObject(object):
	def __init__(self, ):
		super(SaveableObject, self).__init__()
		self.a = None

	def set_a(self, a):
		self.a = a

	def save_object(self,fullpath):
		output = open(fullpath, 'wb')
		pickle.dump(self, output)
		output.close()

	def load_object(self, fullpath):
		pkl_file = open(fullpath, 'rb')
		this_loaded_object = pickle.load(pkl_file)
		pkl_file.close() 
		return this_loaded_object

def intersect_fraction_a_in_b(a,b):
	'''
	a, b binary np.array of the same size with dtype float
	Motivation: 
	  if b is proper subset of a, then output is 1
	  if a is proper subset of b, then sum(b intersect a)/sum(b) 
	b is the limiting factor

	a = np.random.randint(0,2,(4,4)).astype(np.float)
	b = np.random.randint(0,2,(4,4)).astype(np.float)

	m = intersect_fraction_a_in_b(a,b)
	print(a, sum(a.reshape(-1)))
	print(b, sum(b.reshape(-1)))
	print()
	print(a*b)
	print(m)
	'''
	y = a*b
	return sum(y.reshape(-1))/sum(b.reshape(-1))