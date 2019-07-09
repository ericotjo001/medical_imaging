from multivutils import *

class SliceMaker(object):
	"""Abstract Class SliceMaker
	"""
	def __init__(self, full_shape):
		super(SliceMaker, self).__init__()
		self.full_shape = full_shape
		self.slice_shape = None
		self.slice_collection = []
		return

	def set_slice_shape(self): pass

	def get_uniform_slices_no_padding(self): pass

	def print_general_info(self):
		print('self.full_shape  : %s'%(str(self.full_shape)))
		print('self.slice_shape : %s'%(str(self.slice_shape)))

	def reconstruct_uniform_slices_no_padding(self, reconstruction_mapping):
		'''
		params:
		  reconstruction_mapping. List [x1,...,xN] where xk = (slice_k, img_slice_k)
		  where slice_k is the 3D slice object, img_slice_k 3D numpy array slice of the 
		  original ND image
		'''

		reconstructed = np.zeros(shape=self.full_shape)
		for x in reconstruction_mapping:
			reconstructed[x[0]] = x[1]	
		return reconstructed	

class Sampler(object):
	"""Sampler abstract class"""
	def __init__(self, full_shape, slice_shape):
		super(Sampler, self).__init__()
		self.full_shape = full_shape
		self.slice_shape = slice_shape

	def get_center(self): pass
		


def generate_config(this_dir=None):
	print("Generating configuration file...")
	CONFIG_FILE = {
		'test': {
			'multiv_uniform2D_test':{
				'np.shape'   : "[6, 8]",
				'slice_shape': "[2, 3]"},
			'multiv_uniform3D_test':{
				'np.shape'   : "[5, 6, 8]",
				'slice_shape': "[2, 2, 3]"},
			'multiv_unirand2D_test':{
				'np.shape'   : "[6, 8]",
				'slice_shape': "[[0,1];[1,1]]",
				'n_sample'   : "10",
				'save_static_figures': "1",
				'save_gif'			 : "1"},
			'multiv_unirand3D_test':{
				'np.shape'   : "[6, 8, 5]",
				'slice_shape': "[[1,0];[1,1];[1,1]]",
				'n_sample'   : "10",
				'save_static_figures': "1",
				'save_gif'			 : "1"},
			'dualview2D_test':{
				'np.shape'   : "[6, 8]",
				'slice_shape': "[[0,1];[1,1]]",
				'shell_shape': "[[1,1];[2,1]]",
				'n_sample'   : "10",
				'save_static_figures': "1",
				'save_gif'			 : "1"},
			'dualview3D_test':{
				'np.shape'   : "[6, 8, 5]",
				'slice_shape': "[[1,0];[1,1];[1,1]]",
				'shell_shape': "[[1,1];[2,1];[2,2]]",
				'n_sample'   : "10",
				'save_static_figures': "1",
				'save_gif'			 : "1"}
			}
	}
	
	if this_dir is None:
		this_dir = 'multivconfig.json'
	with open(this_dir, 'w') as json_file:
		json.dump(CONFIG_FILE, json_file, separators=(',', ': '), indent=2)
	print("Config file generated: ", this_dir)
	return


class ConfigParser(object):
	"""ConfigParser"""
	def __init__(self, this_dir, json_headers):
		"""
		params:
		  this_dir = directory to the json config file
		  json_headers. list. [key1, key2, ...]. The key corresponds to the subsequent keys
		    to access in the CONFIG FILE, for example ['test', 'multiv_uniform2D_test']
		"""
		super(ConfigParser, self).__init__()
		self.config_raw_data = json_to_dict(this_dir)
		self.keys = json_headers
		return 
		
	def get_config_data(self, verbose=10):
		if verbose > 9: print("ConfigParser.get_config_data()")
		if self.keys[0] == 'test':
			if self.keys[1] == 'multiv_uniform2D_test' or self.keys[1] == 'multiv_uniform3D_test':
				shapeND, slice_shape = self.get00001(verbose=verbose)
				return shapeND, slice_shape
			elif self.keys[1] == 'multiv_unirand2D_test' or self.keys[1] == 'multiv_unirand3D_test':
				data_dict = self.get00004(verbose=verbose)
				return data_dict
			elif self.keys[1] == 'dualview2D_test' or self.keys[1] == 'dualview3D_test':
				data_dict = self.get00004(with_shell=True, verbose=verbose)
				return data_dict		
		else:
			raise Exception("ConfigParser. Error at get_config_data()")
		return

	

	"""
	Auxiliary functions
	"""

	def get00001(self, verbose=10):
		shapeND = self.config_raw_data[self.keys[0]][self.keys[1]]['np.shape']
		shapeND = sanitize_json_strings(shapeND, mode='one_level_list', submode='output_numpy_float_array', verbose=0)
		shapeND = [int(x) for x in shapeND]

		slice_shape = self.config_raw_data[self.keys[0]][self.keys[1]]['slice_shape']
		slice_shape = sanitize_json_strings(slice_shape, mode='one_level_list', submode='output_numpy_float_array', verbose=0)
		slice_shape = [int(x) for x in slice_shape]
		if verbose > 9: 
			print("  shapeND:     ", shapeND)
			print("  slice_shape: ", slice_shape)
		return shapeND, slice_shape

	def get00002(self):
		shapeND = self.config_raw_data[self.keys[0]][self.keys[1]]['np.shape']
		shapeND = sanitize_json_strings(shapeND, mode='one_level_list', submode='output_numpy_float_array', verbose=0)
		shapeND = [int(x) for x in shapeND]

		slice_shape = self.config_raw_data[self.keys[0]][self.keys[1]]['slice_shape']
		slice_shape = sanitize_json_strings(slice_shape, mode='two_level_list', submode='output_numpy_float_array', verbose=0)
		return shapeND, slice_shape

	def get00002_1(self):
		shell_shape = self.config_raw_data[self.keys[0]][self.keys[1]]['shell_shape']
		shell_shape = sanitize_json_strings(shell_shape, mode='two_level_list', submode='output_numpy_float_array', verbose=0)
		return shell_shape

	def get00003_figure_toggles(self):
		save_gif = self.config_raw_data[self.keys[0]][self.keys[1]]["save_gif"]
		save_static_figures = self.config_raw_data[self.keys[0]][self.keys[1]]['save_static_figures']
		return save_gif, save_static_figures

	def get00004(self, with_shell=False, verbose=10):
		shapeND, slice_shape = self.get00002()	
		data_dict = {}
		data_dict['shapeND'] = shapeND 
		data_dict['slice_shape'] = slice_shape 
		if with_shell:
			shell_shape = self.get00002_1()
			data_dict['shell_shape'] = shell_shape
		data_dict['n_sample'] = int(self.config_raw_data[self.keys[0]][self.keys[1]]['n_sample'])
		save_gif, save_static_figures = self.get00003_figure_toggles()
		data_dict['save_static_figures'] = bool(int(save_static_figures))
		data_dict['save_gif'] = bool(int(save_gif))
		if verbose > 9:
			for this_key in data_dict: print("  ", this_key, ":" , data_dict[this_key])
		return data_dict

