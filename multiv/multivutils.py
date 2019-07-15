import os, argparse, sys, json, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
pip install pillow
pip install matplotlib

Optional:
pip install imageio
"""

def find_current_folder(this_path):
	temp=this_path[:]
	while True:
		poss=temp.find("\\")
		if poss is -1:
			break
		temp=temp[poss+1:]
	return temp

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



dir_path=os.getcwd()
this_folder=find_current_folder(dir_path)
parent_folder=dir_path[:len(dir_path)-len(this_folder)-1]
sys.path.append(parent_folder)


DESCRIPTION = '''
\t\t\t=========================================
\t\t\t===    Welcome to multiv package!     ===
\t\t\t=========================================
This package is initially intended to sample slices off 3D images for network training.


\t\t\t=========================================
\t\t\t===               Mode                ===
\t\t\t=========================================

generate_config
  Create a default config file. 
  Optional arg:
    dir. Full path to the configuration file. Default multivconfig.json in the working dir.
  Example usage:
  python multiv.py --mode generate_config
  python multiv.py --mode generate_config --dir path/to/multivconfig.json

test
  These tests are supplied as demonstration of the functions' usage.
  Refer to multivtests.py or the tutorials for details of implementation.
  Submode
  (1) multiv_test1 (Default). 
    No function.
    Example usage:
    python multiv.py --mode test --submode multiv_test1
  (2) multiv_uniform2D_test. 
    This test shows how Slice2D object is used to slice 2D object uniformly.
    Example usage:
    python multiv.py --mode test --submode multiv_uniform2D_test
  (3) multiv_uniform3D_test. 
    Same as uniform2D but for 3D.
    Example usage:
    python multiv.py --mode test --submode multiv_uniform3D_test
  (4) multiv_unirand2D_test. 
    This test shows how Sampler2D object is used to sample 2D slices uniformly.
    Example usage:
    python multiv.py --mode test --submode multiv_unirand2D_test
  (5) multiv_unirand3D_test. 
    This test shows how Sampler3D object is used to sample 3D slices uniformly.
    Example usage:
    python multiv.py --mode test --submode multiv_unirand3D_test
  (6) dualview2D_test
    This test shows how MultiViewSampler2D object is used to sample 2D local and 
    "global" slices uniformly
    Example usage:
    python multiv.py --mode test --submode dualview2D_test
  (7) dualview3D_test
    This test shows how MultiViewSampler3D object is used to sample 3D local and 
    "global" slices uniformly
    Example usage:
    python multiv.py --mode test --submode dualview3D_test
  (8) dualuniform2D_test
    This test shows how 2D array is uniformly sliced, but with envelop.
    Example usage:
    python multiv.py --mode test --submode dualuniform2D_test
  (9) dualuniform3D_test
    This test shows how 3D array is uniformly sliced, but with envelop.
    Example usage:
    python multiv.py --mode test --submode dualuniform3D_test
'''


