from multivh2 import *

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