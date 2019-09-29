from utils.debug_switches import *

import os, argparse, sys, json, time, pickle, time, csv, collections, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from PIL import Image
from multiprocessing import Pool
from os import listdir

import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

this_device = torch.device('cuda:0')
# number_of_classes = 2 # 0,1

DESCRIPTION = '''\t\t\t=== Welcome to imgpro/main.py! ===

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

(X) shorcut_sequence
  python main.py --mode shortcut_sequence
  python main.py --mode shortcut_sequence --config_dir config.json
'''


CONFIG_FILE = {
	'working_dir':"D:/Desktop@D/meim2venv/imgpro",
	'relative_checkpoint_dir':'checkpoints',
	'data_directory':{
		'cifar10': 'D:/Desktop@D/meim2venv/imgpro/data/cifar-10-batches-py'
	},
	'data_submode':'load_cifar10_type0001',
	'model_label_name': 'SmallCNN_XXXXXX',
	'training_mode': 'training_sr_small_cnn',
	'evaluation_mode': 'evaluate_small_cnn',
	# 'lrp_mode': 'lrp_UNet3D_overfit_visualizer',
	'debug_test_mode':"test_load_cifar",
	
	'basic':{
		'batch_size' : 4,
		'n_epoch' : 2,
		'save_model_every_N_epoch': 1,
		'keep_at_most_n_latest_models':24
	},
	'learning':{
		'mechanism':"adam",
		'momentum':0.9,
		'learning_rate':0.0002,
		'weight_decay':0.00001,
		'betas':[0.5,0.9],
		},
	'learning_for_discriminator':{
		'mechanism':"adam",
		'momentum':0.9,
		'learning_rate':0.0002,
		'weight_decay':0.00001,
		'betas':[0.5,0.9],
		},

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
		config_data = config_data._asdict()
		for xkey in config_data:
			if 'ConfigX' in str(type(config_data[xkey])) :
				config_data[xkey] = config_data[xkey]._asdict()
		return config_data


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

def print_info():
	print(DESCRIPTION)

def create_config_file(config_dir):
	cm = ConfigManager()
	cm.create_config_file(config_dir)

	config_data = cm.json_file_to_pyobj(config_dir)
	cm.json_file_to_pyobj_recursive_print(config_data, name='config_data', verbose=0)
	config_data = cm.recursive_namedtuple_to_dict(config_data)

def get_config_data(config_dir):
	cm = ConfigManager()
	config_data = cm.json_file_to_pyobj(config_dir) # it is now a named tuple
	config_data = cm.recursive_namedtuple_to_dict(config_data)
	return config_data