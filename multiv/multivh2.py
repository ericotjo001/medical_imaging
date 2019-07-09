from multivh import *

class Sampler2D(Sampler):
	""" Sampler2D 
	full_shape: list or tuples [nx, ny], (nx, ny) where nx, ny are integers
	slice_shape: [[nx_lower, nx_upper], [ny_lower, ny_upper]]
	  If the center is (cx, cy), then the slice takes the range of cx - nx_lower
	  to cx + nx_upper and likewise for y. Hence,there are nx_lower + nx_upper + 1
	  pixels in the slice along the x direction, and similarly for y. 
	
	"""
	def __init__(self, full_shape, slice_shape):
		super(Sampler2D, self).__init__(full_shape, slice_shape)
		self.cx_range = [0, full_shape[0]] # x center index range for random sampling
		self.cy_range = [0, full_shape[1]]
	
	def set_center_range(self, padding=None, verbose=0):
		if padding is None:
			nxl, nxu = self.slice_shape[0][0], self.slice_shape[0][1]
			nyl, nyu = self.slice_shape[1][0], self.slice_shape[1][1]
			self.cx_range = [nxl, self.full_shape[0] - nxu]
			self.cy_range = [nyl, self.full_shape[1] - nyu]
			if verbose > 9: print("  self.cx_range, self.cy_range: ",self.cx_range, ",", self.cy_range)
	
	def get_center(self):
		this_center_x = np.random.randint(self.cx_range[0], self.cx_range[1])
		this_center_y = np.random.randint(self.cy_range[0], self.cy_range[1])
		return this_center_x, this_center_y

	def get_slice(self, slice_center):
		this_center_x, this_center_y = slice_center	
		xlower, xupper = this_center_x - self.slice_shape[0][0], this_center_x + self.slice_shape[0][1]
		ylower, yupper = this_center_y - self.slice_shape[1][0], this_center_y + self.slice_shape[1][1] 
		""" plus 1 is needed since slice upper limit is exclusive, while the lower limit is inclusive
		"""
		xslice = slice(int(xlower), int(xupper) + 1, None)
		yslice = slice(int(ylower), int(yupper) + 1, None)
		return (xslice, yslice)

	def test_range(self, n_tries=1000, verbose=0):	
		''' Using the ranges set, try to extract slices from the full-shape object.
		Each slice is is defined by slice_shape.
		This test ensures that we are extracting the whole available region by checking that 
		the index 0 and the final index are reached (which set check variable's entries to ones).
		'''
		x = np.zeros(shape=self.full_shape)
		check = [0,0,0,0]
		if verbose > 19: print("\n    %-3s | %-3s | %-3s | %-3s"%('xl','xu','yl','yu'))
		for i in range(n_tries):
			this_center_x, this_center_y = self.get_center()	
			xlower, xupper = this_center_x - self.slice_shape[0][0], this_center_x + self.slice_shape[0][1]
			ylower, yupper = this_center_y - self.slice_shape[1][0], this_center_y + self.slice_shape[1][1] 
			if verbose > 9: print("    %-3s | %-3s | %-3s | %-3s"%(str(int(xlower)),str(int(xupper)),str(int(ylower)),str(int(yupper))))
			if xlower == 0: check[0] = 1
			if xupper == self.full_shape[0] - 1: check[1] = 1
			if ylower == 0: check[2] = 1
			if yupper == self.full_shape[1] - 1: check[3] = 1
		check_result = np.all(check == [1, 1, 1, 1])
		return check_result
			

class Sampler3D(Sampler):
	"""Sampler3D"""
	def __init__(self, full_shape, slice_shape):
		super(Sampler3D, self).__init__(full_shape, slice_shape)
		self.cx_range = [0, full_shape[0]] # x center index range for random sampling
		self.cy_range = [0, full_shape[1]]
		self.cz_range = [0, full_shape[2]]
		
	def set_center_range(self, padding=None, verbose=0):
		if padding is None:
			nxl, nxu = self.slice_shape[0][0], self.slice_shape[0][1]
			nyl, nyu = self.slice_shape[1][0], self.slice_shape[1][1]
			nzl, nzu = self.slice_shape[2][0], self.slice_shape[2][1]
			self.cx_range = [nxl, self.full_shape[0] - nxu]
			self.cy_range = [nyl, self.full_shape[1] - nyu]
			self.cz_range = [nzl, self.full_shape[2] - nzu]
			if verbose > 9: print("  self.cx_range, self.cy_range, self.cz_range: ",self.cx_range, ",", self.cy_range, ",", self.cz_range)
	
	def get_slice(self, slice_center):
		this_center_x, this_center_y, this_center_z = slice_center	
		xlower, xupper = this_center_x - self.slice_shape[0][0], this_center_x + self.slice_shape[0][1]
		ylower, yupper = this_center_y - self.slice_shape[1][0], this_center_y + self.slice_shape[1][1] 
		zlower, zupper = this_center_z - self.slice_shape[2][0], this_center_z + self.slice_shape[2][1] 
		""" plus 1 is needed since slice upper limit is exclusive, while the lower limit is inclusive
		"""
		xslice = slice(int(xlower), int(xupper) + 1, None)
		yslice = slice(int(ylower), int(yupper) + 1, None)
		zslice = slice(int(zlower), int(zupper) + 1, None)
		return (xslice, yslice,zslice)
	
	def get_center(self):
		this_center_x = np.random.randint(self.cx_range[0], self.cx_range[1])
		this_center_y = np.random.randint(self.cy_range[0], self.cy_range[1])
		this_center_z = np.random.randint(self.cz_range[0], self.cz_range[1])
		return this_center_x, this_center_y, this_center_z

	def test_range(self, n_tries=1000, verbose=0):	
		''' Using the ranges set, try to extract slices from the full-shape object.
		Add or minus 1 to the ranges to check that error SHOULD occur to ensure
		that we are extracting the whole available region.
		'''
		x = np.zeros(shape=self.full_shape)
		check = [0, 0, 0, 0, 0, 0]
		if verbose > 19: print("\n    %-3s | %-3s | %-3s | %-3s | %-3s | %-3s"%('xl','xu','yl','yu','zl','zu'))
		for i in range(n_tries):
			this_center_x, this_center_y, this_center_z = self.get_center()	
			xlower, xupper = this_center_x - self.slice_shape[0][0], this_center_x + self.slice_shape[0][1]
			ylower, yupper = this_center_y - self.slice_shape[1][0], this_center_y + self.slice_shape[1][1] 
			zlower, zupper = this_center_z - self.slice_shape[2][0], this_center_z + self.slice_shape[2][1] 
			if verbose > 9: print("    %-3s | %-3s | %-3s | %-3s | %-3s | %-3s"%(str(int(xlower)),str(int(xupper)),str(int(ylower)),str(int(yupper)),str(int(zlower)),str(int(zupper))))
			if xlower == 0: check[0] = 1
			if xupper == self.full_shape[0] - 1: check[1] = 1
			if ylower == 0: check[2] = 1
			if yupper == self.full_shape[1] - 1: check[3] = 1
			if zlower == 0: check[4] = 1
			if zupper == self.full_shape[2] - 1: check[5] = 1
		check_result = np.all(check == [1, 1, 1, 1, 1, 1])
		return check_result

class Slice2D(SliceMaker):
	"""Slice2D"""
	# def __init__(self, full_shape):
	# 	super(Slice2D, self).__init__()
	# 	return

	def set_slice_shape(self, slice_shape):
		''' Set the shape of each slice
		params:
		  slice_shape: list [nx, ny] where nx, ny are integers
		'''
		assert(len(slice_shape) == 2)
		self.slice_shape = slice_shape

	def get_uniform_slices_no_padding(self):
		i,j = 0, 0
		x_upper, x_inter, y_upper, y_inter = self.full_shape[0], self.slice_shape[0], self.full_shape[1], self.slice_shape[1]
		while i < x_upper:
			while j < y_upper:
				i_next, j_next = i + x_inter, j + y_inter
				if i_next > x_upper: i_next = x_upper
				if j_next > y_upper: j_next = y_upper 
				this_slice = (slice(i, i_next, None), slice(j, j_next, None))
				self.slice_collection.append(this_slice)
				j = j + y_inter
			i, j = i + x_inter, 0
		return


class Slice3D(SliceMaker):
	def set_slice_shape(self, slice_shape):
		''' Set the shape of each slice
		params:
		  slice_shape: list [nx, ny, nz] where nx, ny, nz are integers
		'''
		assert(len(slice_shape) == 3)
		self.slice_shape = slice_shape

	def get_uniform_slices_no_padding(self):
		i, j, k = 0, 0, 0
		x_upper, x_inter, y_upper, y_inter = self.full_shape[0], self.slice_shape[0], self.full_shape[1], self.slice_shape[1]
		z_upper, z_inter = self.full_shape[2], self.slice_shape[2]
		while i < x_upper:
			while j < y_upper:
				while k < z_upper:
					i_next, j_next, k_next = i + x_inter, j + y_inter, k + z_inter
					if i_next > x_upper: i_next = x_upper
					if j_next > y_upper: j_next = y_upper 
					if k_next > z_upper: k_next = z_upper
					this_slice = (slice(i, i_next, None), slice(j, j_next, None), slice(k, k_next, None))
					self.slice_collection.append(this_slice)
					k = k + z_inter
				j, k = j + y_inter, 0
			i, j, k = i + x_inter, 0, 0
		return
			
