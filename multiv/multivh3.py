from multivh2 import *


class MultiViewSlicer2D(Slice2D):
	"""MultiViewSlicer2D"""

	def __init__(self, full_shape):
		super(MultiViewSlicer2D, self).__init__(full_shape)
		self.x = None
		self.x_padded = None
		self.minor_view_slice_collection = []
		self.major_view_slice_collection = []
		# self.slice_collection = [] # inherited

	def multiview_slice_and_dice_zero_padded(self, x, slice_shape, shell_extension, verbose=0):
		'''
		x is padded twice.
		First, x is padded to ensure that each extracted slice has the same shape.
		  To illustrate in 1D, if [1,2,3,4,5] is sliced with shape (2,) it is padded to
		  [1,2,3,4,5,0] so that the slice is [1,2], [3,4], [5,0]
		The second padding is to account for the shape of the shell. In the same 1D example, 
		  if the shell has shape[[1,2]], the additional padding to [1,2,3,4,5,0] now makes it
		  [0, 1,2,3,4,5,0, 0,0] 

		params:
		  x: 2d array object whose slices and shells are to be extracted
		  slice_shape: tuple (nx,ny), the shape of slices to be extracted from x.
		  shell_extension: tuple (nx,ny), the shape of shells surrounding each shell to be extracted from x.

		'''
		if verbose > 9: print("multiview_slice_and_dice()")

		self.x = x
		self.set_slice_shape(slice_shape)
		self.__compute_standard_padding()
		s, self.full_shape = self.x.shape, self.x.shape

		self.get_uniform_slices_no_padding()
		self.X = self.__get_multiview_padded_obj(self.x, shell_extension)	# exploded view	
		self.get_multi_view_slices(shell_extension)

		if verbose > 19: 
			for this_slice in self.slice_collection: print("  ",this_slice)
			print("  self.x:\n",self.x)
			print("  self.x.shape:%s"%(str(self.x.shape)))
			print("  self.X:\n",self.X)
			print("  self.X.shape:%s"%(str(self.X.shape)))

	def get_multi_view_slices(self, shell_extension):
		for this_slice in self.slice_collection:
			x_slice = this_slice[0]
			y_slice = this_slice[1]
			
			x_slice_major = slice(int(x_slice.start), int(x_slice.stop + shell_extension[0][0] + shell_extension[0][1]), None)
			y_slice_major = slice(int(y_slice.start), int(y_slice.stop + shell_extension[1][0] + shell_extension[1][1]), None)
			self.major_view_slice_collection.append([x_slice_major, y_slice_major])


			''' standard padding '''
			x_slice_minor = slice(int(x_slice.start + shell_extension[0][0]), int(x_slice.stop + shell_extension[0][0]), None) 
			y_slice_minor = slice(int(y_slice.start + shell_extension[1][0]), int(y_slice.stop + shell_extension[1][0]), None) 
			self.minor_view_slice_collection.append([x_slice_minor, y_slice_minor])
			
	def __compute_standard_padding(self):
		x_slice_shape = self.slice_shape[0]
		x_full_shape = self.full_shape[0]
		y_slice_shape = self.slice_shape[1]
		y_full_shape = self.full_shape[1]

		x_padded_shape = np.ceil(x_full_shape/x_slice_shape) * x_slice_shape
		y_padded_shape = np.ceil(y_full_shape/y_slice_shape) * y_slice_shape
		x_shape_diff = int(x_padded_shape - x_full_shape)
		y_shape_diff = int(int(y_padded_shape - y_full_shape))

		''' standard padding '''
		padding_spec = ((0,x_shape_diff), (0, y_shape_diff)) 
	
		self.x = np.pad(self.x, padding_spec, 'constant', constant_values = 0)
		return		
		
	def __get_multiview_padded_obj(self, x, shell_extension):
		return np.pad(x, shell_extension, 'constant', constant_values=0)

	def reconstruct_uniform_slices_with_padding(self, reconstruction_mapping, original_obj_shape, padded_obj_shape, shell_extension):
		'''
		params:
		  reconstruction_mapping. List [x1,...,xN] where xk = (slice_k, img_slice_k)
		    where slice_k is the 3D slice object, img_slice_k 2D numpy array slice of the 
		    original ND image
		  original_obj_shape. tuple (nx, ny). Original shape of the object to be reconstructed.
		  padded_obj_shape. tuple (nx, ny). shape of the object after twice padding (see the description of multiview_slice_and_dice_zero_padded())
		    the shell_extension 
		'''
		reconstructed_temp = self.reconstruct_uniform_slices_no_padding(reconstruction_mapping, shape=padded_obj_shape)
		
		x_slice = slice(shell_extension[0][0],padded_obj_shape[0] - shell_extension[0][1],None)
		y_slice = slice(shell_extension[1][0],padded_obj_shape[1] - shell_extension[1][1],None)
		reconstructed_temp2 = reconstructed_temp[(x_slice,y_slice)]

		''' standard padding (see __compute_standard_padding()'''
		x_slice = slice(0, original_obj_shape[0], None)
		y_slice = slice(0, original_obj_shape[1], None)
		reconstructed = reconstructed_temp2[(x_slice, y_slice)]	
		return reconstructed 

class MultiViewSlicer3D(Slice3D):
	"""MultiViewSampler3D"""
	def __init__(self, full_shape):
		super(MultiViewSlicer3D, self).__init__(full_shape)
		self.x = None
		self.x_padded = None
		self.minor_view_slice_collection = []
		self.major_view_slice_collection = []

	def multiview_slice_and_dice_zero_padded(self, x, slice_shape, shell_extension, verbose=0):
		'''
		x is padded twice.
		First, x is padded to ensure that each extracted slice has the same shape.
		  To illustrate in 1D, if [1,2,3,4,5] is sliced with shape (2,) it is padded to
		  [1,2,3,4,5,0] so that the slice is [1,2], [3,4], [5,0]
		The second padding is to account for the shape of the shell. In the same 1D example, 
		  if the shell has shape[[1,2]], the additional padding to [1,2,3,4,5,0] now makes it
		  [0, 1,2,3,4,5,0, 0,0] 

		params:
		  x: 2d array object whose slices and shells are to be extracted
		  slice_shape: tuple (nx,ny), the shape of slices to be extracted from x.
		  shell_extension: tuple (nx,ny), the shape of shells surrounding each shell to be extracted from x.

		'''
		if verbose > 9: print("multiview_slice_and_dice()")

		self.x = x
		self.set_slice_shape(slice_shape)
		self.__compute_standard_padding()
		s, self.full_shape = self.x.shape, self.x.shape

		self.get_uniform_slices_no_padding()
		self.X = self.__get_multiview_padded_obj(self.x, shell_extension)	# exploded view	
		self.get_multi_view_slices(shell_extension)

		if verbose > 19: 
			for this_slice in self.slice_collection: print("  ",this_slice)
			print("  self.x:\n",self.x)
			print("  self.x.shape:%s"%(str(self.x.shape)))
			print("  self.X:\n",self.X)
			print("  self.X.shape:%s"%(str(self.X.shape)))

	def get_multi_view_slices(self, shell_extension):
		for this_slice in self.slice_collection:
			x_slice = this_slice[0]
			y_slice = this_slice[1]
			z_slice = this_slice[2]
			
			x_slice_major = slice(int(x_slice.start), int(x_slice.stop + shell_extension[0][0] + shell_extension[0][1]), None)
			y_slice_major = slice(int(y_slice.start), int(y_slice.stop + shell_extension[1][0] + shell_extension[1][1]), None)
			z_slice_major = slice(int(z_slice.start), int(z_slice.stop + shell_extension[2][0] + shell_extension[2][1]), None)
			self.major_view_slice_collection.append([x_slice_major, y_slice_major, z_slice_major])

			''' standard padding '''
			x_slice_minor = slice(int(x_slice.start + shell_extension[0][0]), int(x_slice.stop + shell_extension[0][0]), None) 
			y_slice_minor = slice(int(y_slice.start + shell_extension[1][0]), int(y_slice.stop + shell_extension[1][0]), None) 
			z_slice_minor = slice(int(z_slice.start + shell_extension[2][0]), int(z_slice.stop + shell_extension[2][0]), None) 
			self.minor_view_slice_collection.append([x_slice_minor, y_slice_minor, z_slice_minor])
			
	def __compute_standard_padding(self):
		x_slice_shape = self.slice_shape[0]
		x_full_shape = self.full_shape[0]
		y_slice_shape = self.slice_shape[1]
		y_full_shape = self.full_shape[1]
		z_slice_shape = self.slice_shape[2]
		z_full_shape = self.full_shape[2]

		x_padded_shape = np.ceil(x_full_shape/x_slice_shape) * x_slice_shape
		y_padded_shape = np.ceil(y_full_shape/y_slice_shape) * y_slice_shape
		z_padded_shape = np.ceil(z_full_shape/z_slice_shape) * z_slice_shape
		x_shape_diff = int(x_padded_shape - x_full_shape)
		y_shape_diff = int(int(y_padded_shape - y_full_shape))
		z_shape_diff = int(int(z_padded_shape - z_full_shape))
		''' standard padding '''
		padding_spec = ((0,x_shape_diff), (0, y_shape_diff), (0, z_shape_diff)) 
		
		self.x = np.pad(self.x, padding_spec, 'constant', constant_values = 0)
		return	
			
	def __get_multiview_padded_obj(self, x, shell_extension):
		return np.pad(x, shell_extension, 'constant', constant_values=0)

	def reconstruct_uniform_slices_with_padding(self, reconstruction_mapping, original_obj_shape, padded_obj_shape, shell_extension):
		'''
		params:
		  reconstruction_mapping. List [x1,...,xN] where xk = (slice_k, img_slice_k)
		    where slice_k is the 3D slice object, img_slice_k 3D numpy array slice of the 
		    original ND image
		  original_obj_shape. tuple (nx, ny, nz). Original shape of the object to be reconstructed.
		  padded_obj_shape. tuple (nx, ny, nz). shape of the object after twice padding (see the description of multiview_slice_and_dice_zero_padded())
		    the shell_extension 
		'''
		reconstructed_temp = self.reconstruct_uniform_slices_no_padding(reconstruction_mapping, shape=padded_obj_shape)
		
		x_slice = slice(shell_extension[0][0],padded_obj_shape[0] - shell_extension[0][1],None)
		y_slice = slice(shell_extension[1][0],padded_obj_shape[1] - shell_extension[1][1],None)
		z_slice = slice(shell_extension[2][0],padded_obj_shape[2] - shell_extension[2][1],None)
		reconstructed_temp2 = reconstructed_temp[(x_slice, y_slice, z_slice)]

		''' standard padding (see __compute_standard_padding()'''
		x_slice = slice(0, original_obj_shape[0], None)
		y_slice = slice(0, original_obj_shape[1], None)
		z_slice = slice(0, original_obj_shape[2], None)
		reconstructed = reconstructed_temp2[(x_slice, y_slice, z_slice)]	
		return reconstructed 

class MultiViewSampler2D(Sampler2D):
	"""MultiViewSampler2D"""
	def __init__(self, full_shape, slice_shape, shell_shape):
		super(MultiViewSampler2D, self).__init__(full_shape, slice_shape)
		self.assert_shell_shape_is_larger(shell_shape)
		self.shell_shape = shell_shape

	def assert_shell_shape_is_larger(self, shell_shape):
		assert(np.all(shell_shape[0]>=self.slice_shape[0]))
		assert(np.all(shell_shape[1]>=self.slice_shape[1]))

	def set_center_range(self, padding=None, verbose=0):
		if padding is None:
			nxl, nxu = self.shell_shape[0][0], self.shell_shape[0][1]
			nyl, nyu = self.shell_shape[1][0], self.shell_shape[1][1]
			self.cx_range = [nxl, self.full_shape[0] - nxu]
			self.cy_range = [nyl, self.full_shape[1] - nyu]
			if verbose > 9: print("  self.cx_range, self.cy_range: ",self.cx_range, ",", self.cy_range)

	def get_shell_slice(self, slice_center):
		this_center_x, this_center_y = slice_center	
		xlower, xupper = this_center_x - self.shell_shape[0][0], this_center_x + self.shell_shape[0][1]
		ylower, yupper = this_center_y - self.shell_shape[1][0], this_center_y + self.shell_shape[1][1] 
		""" plus 1 is needed since slice upper limit is exclusive, while the lower limit is inclusive
		"""
		xslice = slice(int(xlower), int(xupper) + 1, None)
		yslice = slice(int(ylower), int(yupper) + 1, None)
		return (xslice, yslice)

	def test_range(self, n_tries=100, verbose=0):	
		''' Using the ranges set, try to extract slices from the full-shape object.
		Each slice is wrapped in a shell whose shape is defined by shell_shape.
		This test ensures that we are extracting the whole available region by checking that 
		the index 0 and the final index are reached (which set check variable's entries to ones).
		If the indices exceed max index or go below 0, error will occur subsequently, but not checked here.
		'''
		x = np.zeros(shape=self.full_shape)
		check = [0,0,0,0]
		if verbose > 19: print("\n    %-3s | %-3s | %-3s | %-3s"%('xl','xu','yl','yu'))
		for i in range(n_tries):
			this_center_x, this_center_y = self.get_center()	
			xlower, xupper = this_center_x - self.shell_shape[0][0], this_center_x + self.shell_shape[0][1]
			ylower, yupper = this_center_y - self.shell_shape[1][0], this_center_y + self.shell_shape[1][1] 
			if verbose > 9: print("    %-3s | %-3s | %-3s | %-3s"%(str(int(xlower)),str(int(xupper)),str(int(ylower)),str(int(yupper))))
			if xlower == 0: check[0] = 1
			if xupper == self.full_shape[0] - 1: check[1] = 1
			if ylower == 0: check[2] = 1
			if yupper == self.full_shape[1] - 1: check[3] = 1
		check_result = np.all(check == [1, 1, 1, 1])
		return check_result

class MultiViewSampler3D(Sampler3D):
	"""MultiViewSampler3D"""
	def __init__(self, full_shape, slice_shape, shell_shape):
		super(MultiViewSampler3D, self).__init__(full_shape, slice_shape)
		self.assert_shell_shape_is_larger(shell_shape)
		self.shell_shape = shell_shape
	
	def assert_shell_shape_is_larger(self, shell_shape):
		assert(np.all(shell_shape[0]>=self.slice_shape[0]))
		assert(np.all(shell_shape[1]>=self.slice_shape[1]))
		assert(np.all(shell_shape[2]>=self.slice_shape[2]))

	def get_shell_slice(self, slice_center):
		this_center_x, this_center_y, this_center_z = slice_center	
		xlower, xupper = this_center_x - self.shell_shape[0][0], this_center_x + self.shell_shape[0][1]
		ylower, yupper = this_center_y - self.shell_shape[1][0], this_center_y + self.shell_shape[1][1] 
		zlower, zupper = this_center_z - self.shell_shape[2][0], this_center_z + self.shell_shape[2][1] 
		""" plus 1 is needed since slice upper limit is exclusive, while the lower limit is inclusive
		"""
		xslice = slice(int(xlower), int(xupper) + 1, None)
		yslice = slice(int(ylower), int(yupper) + 1, None)
		zslice = slice(int(zlower), int(zupper) + 1, None)
		return (xslice, yslice,zslice)

	def set_center_range(self, padding=None, verbose=0):
		if padding is None:
			nxl, nxu = self.shell_shape[0][0], self.shell_shape[0][1]
			nyl, nyu = self.shell_shape[1][0], self.shell_shape[1][1]
			nzl, nzu = self.shell_shape[2][0], self.shell_shape[2][1]
			self.cx_range = [nxl, self.full_shape[0] - nxu]
			self.cy_range = [nyl, self.full_shape[1] - nyu]
			self.cz_range = [nzl, self.full_shape[2] - nzu]
			if verbose > 9: print("  self.cx_range, self.cy_range, self.cz_range: ",self.cx_range, ",", self.cy_range, ",", self.cz_range)

		
	def test_range(self, n_tries=1000, verbose=0):	
		''' See MultiViewSampler2D test_range(), similar '''
		x = np.zeros(shape=self.full_shape)
		check = [0, 0, 0, 0, 0, 0]
		if verbose > 19: print("\n    %-3s | %-3s | %-3s | %-3s | %-3s | %-3s"%('xl','xu','yl','yu','zl','zu'))
		for i in range(n_tries):
			this_center_x, this_center_y, this_center_z = self.get_center()	
			xlower, xupper = this_center_x - self.shell_shape[0][0], this_center_x + self.shell_shape[0][1]
			ylower, yupper = this_center_y - self.shell_shape[1][0], this_center_y + self.shell_shape[1][1] 
			zlower, zupper = this_center_z - self.shell_shape[2][0], this_center_z + self.shell_shape[2][1] 
			if verbose > 9: print("    %-3s | %-3s | %-3s | %-3s | %-3s | %-3s"%(str(int(xlower)),str(int(xupper)),str(int(ylower)),str(int(yupper)),str(int(zlower)),str(int(zupper))))
			if xlower == 0: check[0] = 1
			if xupper == self.full_shape[0] - 1: check[1] = 1
			if ylower == 0: check[2] = 1
			if yupper == self.full_shape[1] - 1: check[3] = 1
			if zlower == 0: check[4] = 1
			if zupper == self.full_shape[2] - 1: check[5] = 1
		check_result = np.all(check == [1, 1, 1, 1, 1, 1])
		return check_result	