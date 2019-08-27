import os, argparse, sys, json, time, pickle, time, csv
import numpy as np
import nibabel as nib
import numpy as np

from PIL import Image
from multiprocessing import Pool
from os import listdir

import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

this_device = torch.device('cuda:0')
number_of_classes = 2 # 0,1

DESCRIPTION = '''\t\t\t=== Welcome to meim3/main.py! ===

Implementations of 3D versions of several neural network to handle ISLES 2017 Ischemic Stroke Lesion Segmentation.

Warning: for CONFIG_FILE, check the relevance of the input. For example, in one particular
  instance of our implementation, when learning mechanism is adam, momentum is not used. When 
  SGD is used as the learning mechanism, betas are not relevant.

Modes:
(1) info
  python main.py
  python main.py info

(2) create_config_file
  python main.py --mode create_config_file
  python main.py --mode create_config_file --config_dir config.json 

(3) test
  python main.py --mode test
  python main.py --mode test --config_dir config.json 

(4) train
  python main.py --mode train
  python main.py --mode train --config_dir config.json

(5) evaluation
  python main.py --mode evaluation
  python main.py --mode evaluation --config_dir config.json

Configurations:
The following can be found in the configuration file.
(1) training_mode. See train() [entry.py].
(2) evaluation. See evaluation() [entry.py].
(3) debug_test_mode. See test() [tests/test.py]
(4) augmentation. See custom_augment.py, "section config_data['augmentation']['type']" .
'''

CONFIG_FILE = {
	'working_dir':"D:/Desktop@D/meim2venv/meim3",
	'dir_ISLES2017':"D:/Desktop@D/meim2venv/meim3/data/isles2017",
	'relative_checkpoint_dir':'checkpoints',
	'model_label_name': 'UNet3D_XXXXXX',
	'training_mode': 'UNet3D',
	'evaluation_mode': 'UNet3D_overfit',
	'debug_test_mode':"test_load_many_ISLES2017",
	
	'basic_1':{
		'batch_size' : "2",
		'n_epoch' : "5",
		'save_model_every_N_epoch': "1"
	},
	'dataloader':{
		'resize' : "[192,192,19]"
	},
	'learning':{
		'mechanism':"adam",
		'momentum':"0.9",
		'learning_rate':"0.0002",
		'weight_decay':"0.00001",
		'betas':"[0.5,0.9]",
		},
	'normalization': {
		"ADC_source_min_max": "[None, None]", # "[0, 5000]",
		"MTT_source_min_max": "[None, None]",
		"rCBF_source_min_max": "[None, None]",
		"rCBV_source_min_max": "[None, None]",
		"Tmax_source_min_max": "[None, None]",
		"TTP_source_min_max": "[None, None]",
		
		"ADC_target_min_max": "[0, 1]",
		"MTT_target_min_max": "[0, 1]",
		"rCBF_target_min_max": "[0, 1]",
		"rCBV_target_min_max": "[0, 1]",
		"Tmax_target_min_max": "[0, 1]",
		"TTP_target_min_max": "[0, 1]",
	},
	'augmentation': {
		'type': 'no_augmentation',
		'number_of_data_augmented_per_case': '10',
	}
}
'''
'debug_test_mode'. To see what modes are available, see tests/test.py
'''

def prepare_config(config_raw_data):
	config_data = {}
	config_data['working_dir'] = config_raw_data['working_dir']
	config_data['dir_ISLES2017'] = config_raw_data['dir_ISLES2017']
	config_data['relative_checkpoint_dir'] = config_raw_data['relative_checkpoint_dir']
	config_data['model_label_name'] = config_raw_data['model_label_name']
	config_data['training_mode'] = config_raw_data['training_mode']
	config_data['evaluation_mode'] = config_raw_data['evaluation_mode']
	config_data['debug_test_mode'] = config_raw_data['debug_test_mode']

	config_data['basic_1'] = {}
	config_data['basic_1']['batch_size'] = int(config_raw_data['basic_1']['batch_size'])
	config_data['basic_1']['n_epoch'] = int(config_raw_data['basic_1']['n_epoch'])
	config_data['basic_1']['save_model_every_N_epoch'] = int(config_raw_data['basic_1']['save_model_every_N_epoch'])

	config_data['dataloader'] = {}
	config_data['dataloader']['resize'] = sanitize_json_strings(config_raw_data['dataloader']['resize'], mode='one_level_list', submode='output_numpy_int_array', verbose=0)

	learning = {}
	learning['mechanism'] = config_raw_data['learning']['mechanism']
	learning['momentum'] = float(config_raw_data['learning']['momentum'])
	learning['learning_rate'] = float(config_raw_data['learning']['learning_rate'])
	learning['weight_decay'] = float(config_raw_data['learning']['weight_decay'])
	learning['betas'] = sanitize_json_strings(config_raw_data['learning']['betas'], mode='one_level_list', submode='output_numpy_float_array', verbose=0)
	config_data['learning'] = learning

	normalization = {}
	modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP']
	for mod in modalities_label:
		normalization[mod + '_source_min_max'] = sanitize_json_strings(config_raw_data['normalization'][mod + '_source_min_max'], mode='one_level_list', submode='output_numpy_float_array', verbose=0)
		normalization[mod + '_target_min_max'] = sanitize_json_strings(config_raw_data['normalization'][mod + '_target_min_max'], mode='one_level_list', submode='output_numpy_float_array', verbose=0)
	config_data['normalization'] = normalization

	augmentation = {}
	augmentation['type'] = config_raw_data['augmentation']['type']
	augmentation['number_of_data_augmented_per_case'] = int(config_raw_data['augmentation']['number_of_data_augmented_per_case'])
	config_data['augmentation'] = augmentation
	return config_data

def json_to_dict(json_dir):
	f = open(json_dir)
	fr = f.read()
	data = json.loads(fr)
	f.close()
	return data

def clear_square_brackets(x):
	while True:
		if x.find('[') >=0:
			x = x[1:]
		if x.find(']') >=0:
			x = x[:-1]
		if x.find('[') >=0 or x.find(']') >=0:
			pass
		else:
			break
	return x

def clear_white_spaces_string_front_back(x):
	while True:
		if x[0]==' ': x = x[1:]
		else: break
	x = x[::-1]
	while True:
		if x[0]==' ': x = x[1:]
		else: break
	return x[::-1]

def repeat_split(this_string, delimiter):
	out = []
	while this_string.find(delimiter) > -1:
		temp = this_string.find(delimiter)
		out.append(this_string[:temp])
		this_string = this_string[temp+1:]
	out.append(this_string)
	return out

def sanitize_json_strings(x, mode='one_level_list', submode='output_numpy_float_array', verbose=0):
	"""
	Sanitizing strings, intended for json argument extraction.
	See usage example in each mode
	"""
	if verbose > 9: print("sanitize_json_strings().")
	if mode == 'one_level_list':
		"""
		Example usage:
		x = '[12, 42]'
		x1 = '[ 12, 42]'
		x2 = '[12  , 42   ]'

		for xx in [x, x1, x2]:
			out = sanitize_json_strings(x, verbose = 0)
			print("  ",out)
		"""
		x1 = repeat_split(x, ',')
		if verbose > 199: print("  ",x1)
		temp = []
		for y in x1:
			if verbose > 199: print("  ", clear_square_brackets(y))
			temp.append(clear_square_brackets(y))
		
		return_None = False
		for i,y in enumerate(temp): 
			if clear_white_spaces_string_front_back(y) == "None": temp[i] = None; return_None = True
		if return_None: return temp

		if submode =='output_numpy_float_array': out = np.array([float(y) for y in temp])
		elif submode =='output_numpy_int_array': out = np.array([int(y) for y in temp])
		elif submode =='output_tuple_int': out = tuple([int(y) for y in temp])
		elif submode is None: out = [clear_white_spaces_string_front_back(x) for x in temp]
		return out
	if mode == 'two_level_list':
		"""
		x = '[[12, 42]; [88, 77]]'
		x1 = '[     [  12  , 42]; [  88,   77]]'
		x2 = '[[12, 42]     ;[88,77];[1224,12321, 999]]'

		for xx in [x, x1, x2]:
			out = sanitize_json_strings(xx, mode='two_level_list', submode='output_numpy_float_array', verbose = 200)
			print("  out:", out)
			print("  ",out.shape)
		"""
		x1 = repeat_split(x, ';')
		if verbose > 199: print("  x1:", x1)
		temp = []
		for y in x1:
			if verbose > 199: print("  ", sanitize_json_strings(clear_square_brackets(y), mode='one_level_list', submode=None))
			temp.append(sanitize_json_strings(clear_square_brackets(y), mode='one_level_list', submode=None))
		if submode =='output_numpy_float_array': 
			out = []
			for y in temp: out.append(np.array(y, dtype=np.float))
			out = np.array(out)
		elif submode =='output_numpy_int_array': 
			out = []
			for y in temp: out.append(np.array(y, dtype=np.int))
			out = np.array(out)
		elif submode =='output_tuple_int':
			out = []
			for y in temp: out.append(tuple(np.array(y, dtype=np.int)))
			# out = np.array(out)
		elif submode is None: out = temp
		return out
	return

def printing_config(config_data):
	for x in config_data:
		if not isinstance(config_data[x],dict) :
			print("  %s : %s [%s]"%(x, config_data[x],type(config_data[x])))
		else:
			print("  %s :"%(x))
			for y in config_data[x]:
				print("    %s : %s [%s]"%(y, config_data[x][y], type(config_data[x][y])))

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
	if source_min is None: source_min = np.min(x)
	if source_max is None: source_max = np.max(x)
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