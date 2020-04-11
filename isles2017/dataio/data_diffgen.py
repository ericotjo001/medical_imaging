import numpy as np
import torch
import torch.nn.functional as F
"""
Developed by Erico Tjoa
version 0.1
"""

class DGen(object):
	"""
	Assume 2D is in the form (H,W) i.e. exclude channel. Likewise, 3D is (D,H,W)
	"""
	def __init__(self, unit_size=(28,28)):
		super(DGen, self).__init__()
		self.unit_size = unit_size

	def generate_bundle_binary_tree(self,
		base_directions=(1,-1),
		base_n_steps=10,
		enlarge_range_forward=(2,2), 
		enlarge_range_backward=(2,2),
		n_steps=10,
		directions=(1,1), 
		q_constant=0.01, lower_threshold=0.1, block_factor=0.5,
		init_point='center',
		init_translate=None,
		**kwargs):
		x = np.zeros(shape=self.unit_size)
		s = x.shape
		dim =len(s)

		init_coord = self.get_init_coord(s, init_point=init_point, **kwargs)
		if init_translate is not None:
			init_coord+=init_translate

		seq = self.grow_sequences_of_coordinates_in_singular_direction(init_coord, base_directions, base_n_steps)
		
		output = [x.astype(int) for x in seq]
		output.append(init_coord.astype(int))
		for coord in seq:
			coords_branch = self.grow_sequences_of_coordinates_in_singular_direction(coord, directions, n_steps)
			for new_coord in coords_branch:
				output.append(new_coord.astype(int))
		x = self.get_binary_map_from_coords(output, dim)		
		return x

	def generate_one_seed_binary_tree(self, 
		n_steps=10,
		directions=(1,1), 
		enlarge_range_forward=(2,2), 
		enlarge_range_backward=(2,2),
		q_constant=0.01, lower_threshold=0.1, block_factor=0.5,
		init_point='center',
		init_translate=None,
		**kwargs):
		x = np.zeros(shape=self.unit_size)
		s = x.shape
		dim =len(s)


		init_coord = self.get_init_coord(s, init_point=init_point, **kwargs)
		if init_translate is not None:
			init_coord+=init_translate
		seq = self.grow_sequences_of_coordinates_in_singular_direction(init_coord, directions, n_steps)
		new_set_of_coords = self.enlarge_sequence(seq,enlarge_range_forward, enlarge_range_backward,
			q_constant=q_constant, lower_threshold=lower_threshold, block_factor=block_factor)
		x = self.get_binary_map_from_coords(new_set_of_coords, dim)
		return x 

	def get_init_coord(self, unit_size, init_point='center',**kwargs):
		dim = len(unit_size)
		if init_point == 'center':	
			init_coord = np.zeros(shape=dim)
			for i in range(dim):
				init_coord[i] = np.floor(unit_size[i]/2)
		elif init_point == 'random_center':
			random_center_distance = kwargs.get('random_center_distance')
			init_coord = np.zeros(shape=dim)
			for i in range(dim):
				init_coord[i] = np.floor(unit_size[i]/2) + np.random.randint(-random_center_distance,1+random_center_distance)				
		elif init_point=='manual':
			init_coord = kwargs.get('manual_init_coord')
		else:
			raise Exception('init_point not specified')
		return init_coord

	def get_binary_map_from_coords(self, set_of_coords, dim):
		x = np.zeros(shape=self.unit_size)
		for c in set_of_coords:
			assert(len(c)==dim)
			indices = tuple(c)

			skip_this_coord = False
			for j,this_index in enumerate(indices):
				if this_index>=self.unit_size[j] or this_index<0:
					skip_this_coord = True
			if skip_this_coord: continue
			
			x[indices] = 1.

		return x


	def enlarge_sequence(self, sequence_of_coords, enlarge_range_forward, enlarge_range_backward,
		q_constant=0.1, lower_threshold=0.2, block_factor=1.0):
		# sequence_of_coords is a set of coords (each coord a np.array)
		#   coords can be multi dimensional
		#   if 2D, then example is [4,5]. If 3D then for example [10,10,2]

		dim = len(sequence_of_coords[0])
		# we assume ALL coords have the same dim. To save computation we do not check all

		# enlarge_range forward and backward indicate the size of enlargement (specified in positive integers)
		#   forward refer to the direction along larger indices
		#   backward along smaller indices
		#   for example, if the coord is [5,6], enlarge_range_forward=[4,4], enlarge_range_backward=[1,1]
		#     then new coord generated can be anywhere between [5-1, 6-1] and [5+4, 6+4] (roughly)
		#     see how the extension is actually performed below
		assert(dim==len(enlarge_range_forward) and dim==len(enlarge_range_backward))
		assert(q_constant>=0)

		# block_factor: higher block (nearer 1) smaller growth
		assert(block_factor<=1 and block_factor>=0)

		new_set_of_coords = [coord.astype(int) for coord in sequence_of_coords]
		for this_coord in sequence_of_coords:
			for i_dim, s_range in enumerate(enlarge_range_forward):
				forward_growth_indicator = self.get_growth_indicator(s_range, block_factor, q_constant, lower_threshold)
				for ig in range(sum(forward_growth_indicator)):
					temp = this_coord*0
					temp[i_dim] = ig + 1
					new_set_of_coords.append([x.astype(int) for x in this_coord + temp])
			for i_dim, s_range in enumerate(enlarge_range_backward):
				backward_growth_indicator = self.get_growth_indicator(s_range, block_factor, q_constant, lower_threshold)
				for ig in range(sum(backward_growth_indicator)):
					temp = this_coord*0
					temp[i_dim] = ig + 1
					new_set_of_coords.append([x.astype(int) for x in this_coord - temp])
		return new_set_of_coords

	def get_growth_indicator(self, s_range, block_factor, q_constant, lower_threshold):
		# q_constant: larger q_constant smaller growth
		growth_indicator = []
		for j in range(1,1+s_range):
		 	p = 1./(q_constant+j*block_factor) # probability enlarge the pixel
		 	if p< lower_threshold:
		 		continue
		 	rng = np.random.uniform(0,1)
		 	if rng<p:
		 		growth_indicator = [1 for i in range(len(growth_indicator)+1)]
		 	else:
		 		growth_indicator.append(0)
		return growth_indicator

	def grow_sequences_of_coordinates_in_singular_direction(self,init_coord, directions, n_steps):
		# example init_coord = [5,5] np.array
		# directions = (d1,d2) where d1, d2 are integers
		dim = len(init_coord)
		s = init_coord.shape
		assert(len(init_coord)==len(directions))
		generated_sequence = [init_coord]
		for i in range(n_steps):
			dim_to_grow = np.random.randint(0,dim)

			delta = np.zeros(shape=s)
			# delta[dim_to_grow] += directions[dim_to_grow]
			if directions[dim_to_grow]>=0: 
				delta[dim_to_grow] += np.random.randint(0, 1 + directions[dim_to_grow])
			else:
				delta[dim_to_grow] += np.random.randint(directions[dim_to_grow],0)
			new_coord = generated_sequence[-1] + delta 
			generated_sequence.append(new_coord)
		return generated_sequence

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

class DGenSample(DGen):
	def __init__(self, unit_size=(28,28)):
		super(DGenSample, self).__init__(unit_size = unit_size)
		
	def get_sample(self,n_seeds=480,
		steps=(30,50),
		direction_numbers=(-2,3),
		enlarge_range_forward=(2,2), 
		enlarge_range_backward=(2,2),
		q_constant=0.01, lower_threshold=0.1, block_factor=0.95,
		random_center_distance=30):
		x = np.zeros(shape=self.unit_size)
		for i in range(n_seeds):
			n_steps = np.random.randint(steps[0],steps[1])
			this_direction = tuple([np.random.randint(direction_numbers[0],direction_numbers[1]),
				np.random.randint(direction_numbers[0],direction_numbers[1])])
			x += self.generate_one_seed_binary_tree(
				n_steps=n_steps,
				directions=this_direction, 
				enlarge_range_forward=enlarge_range_forward, 
				enlarge_range_backward=enlarge_range_backward,
				q_constant=q_constant, lower_threshold=lower_threshold, block_factor=block_factor,
				init_point='random_center',
				random_center_distance=random_center_distance)
		x = normalize_numpy_array(x,target_min=0,target_max=1)
		return x

	def get_sample_lesion(self,
		n_seed=(8,10),
		enlarge_range_forward=(2,2), 
		enlarge_range_backward=(1,1),
		q_constant=0.01, lower_threshold=0.1, block_factor=1,
		init_point='random_center',
		random_center_distance=4,
		):
		dim = len(self.unit_size)
		random_translation_distance = (30,30)
		assert(len(random_translation_distance)==dim)
		translation_coord = np.zeros(shape=dim)
		for i in range(dim):
			translation_coord[i] += np.random.randint(-random_translation_distance[i],1+random_translation_distance[i])

		x = np.zeros(shape=self.unit_size)
		rand_n_seed = np.random.randint(n_seed[0],n_seed[1])
		for i in range(rand_n_seed):
			# init_coord = dg.get_init_coord(dg.unit_size, init_point='random_center',
			# 	random_center_distance=40)
			n_steps = np.random.randint(10,20)
			this_direction = tuple([np.random.randint(-2,3),np.random.randint(-2,3)])
			x += self.generate_one_seed_binary_tree(
				n_steps=n_steps,
				directions=this_direction, 
				enlarge_range_forward=enlarge_range_forward, 
				enlarge_range_backward=enlarge_range_backward,
				q_constant=q_constant, lower_threshold=lower_threshold, block_factor=block_factor,
				init_point='random_center',
				random_center_distance=random_center_distance,
				init_translate=translation_coord,
				)
			x = (x>0).astype(np.float)
		x = normalize_numpy_array(x,target_min=0,target_max=1)
		return x

class DG3D(DGenSample):
	"""docstring for DG3D"""
	def __init__(self, unit_size=(28,28), depth=4):
		super(DG3D, self).__init__(unit_size = unit_size)
		self.depth=depth

	def create_3d_from_2d_by_interp(self, x, shape2d=(192,192), this_shape=(19,192,192)):
		x3d = np.zeros(shape=(1,1,3,)+shape2d)
		x3d[0,0,1,:,:] = x # for torch, need (batch,channel, D,H,W)
		r1 = int(np.floor(shape2d[0]/2))
		r2 = int(np.floor(shape2d[1]/2))
		x3d[0,0,0,r1-6:r1+6,r2-6:r2+6] = 0.1+np.random.uniform(0,0.2)
		x3d[0,0,2,r1-6:r1+6,r2-6:r2+6] = 0.1+np.random.uniform(0,0.2)
		x3din = F.interpolate(torch.tensor(x3d),size=this_shape,mode='trilinear',align_corners=True)
		x3din = x3din.squeeze().detach().numpy()
		return x3din

	def generate_data(self, random_mode=False):
		# dg = DGenSample(unit_size=(192,192))
		dim = len(self.unit_size)

		if random_mode:
			x = self.get_sample(n_seeds=480 + np.random.randint(-120,120 + 1),
				steps=(30 - np.random.randint(0,20),50+np.random.randint(0,20)),
				direction_numbers=(-2- np.random.randint(0,3),3+np.random.randint(0,3)),
				enlarge_range_forward=(np.random.randint(2,7),np.random.randint(2,7)), 
				enlarge_range_backward=(np.random.randint(2,7),np.random.randint(2,7)),
				q_constant=np.random.uniform(0.005,0.2), lower_threshold=np.random.uniform(0.1,0.4), block_factor=np.random.uniform(0.5,1),
				random_center_distance=np.random.randint(20,60))
			y = self.get_sample_lesion(n_seed=(6,24),
				enlarge_range_forward=(np.random.randint(2,7),np.random.randint(2,7)), 
				enlarge_range_backward=(np.random.randint(2,7),np.random.randint(2,7)),
				q_constant=np.random.uniform(0.005,0.2), lower_threshold=np.random.uniform(0.1,0.4), block_factor=np.random.uniform(0.5,1),
				init_point='random_center',
				random_center_distance=np.random.randint(1,6))
		else:
			x = self.get_sample()
			y = self.get_sample_lesion()

		x[x>0.8] = 1.
		coin_flip = np.random.uniform(0,1)
		if coin_flip>0.5:
			sgn = -1 
			random_effect_factor = np.random.uniform(0.2,0.8, size=y.shape)
		else:
			sgn = 1
			random_effect_factor = np.random.uniform(0.2,0.8, size=y.shape)
		x1 = np.clip(x +sgn* x*y*random_effect_factor,0,1)
		y= (y-y*(x>0.9).astype(float))

		x3din = self.create_3d_from_2d_by_interp(x,shape2d=x.shape, this_shape=(self.depth,) + self.unit_size)
		x3din_un = self.create_3d_from_2d_by_interp(x1,shape2d=x.shape, this_shape=(self.depth,) + self.unit_size)
		y3din = self.create_3d_from_2d_by_interp(y,shape2d=x.shape, this_shape=(self.depth,) + self.unit_size)
		y3din = (y3din>0.7).astype(float)
		return x3din, x3din_un, y3din

	def generate_data_batches_in_torch(self, channel_size=6, batch_size=2, resize=None):
		from utils.utils import batch_no_channel_interp3d
		x_healthy, x_unhealthy, y_lesion, y_healthy = [],[],[],[]
		for i in range(batch_size):
			x1, xu, yu, y1 = self.generate_multi_channel_data(channel_size=channel_size,random_mode=True)
			x_healthy.append(x1)
			x_unhealthy.append(xu)
			y_lesion.append(yu)
			y_healthy.append(y1)
		x_healthy = torch.tensor(np.stack(x_healthy))
		x_unhealthy = torch.tensor(np.stack(x_unhealthy))
		y_lesion = torch.tensor(np.stack(y_lesion))
		y_healthy= torch.tensor(np.stack(y_healthy))
		if resize is not None:
			x_healthy = F.interpolate(x_healthy, size=resize[::-1],mode='trilinear',align_corners=True)
			x_unhealthy = F.interpolate(x_unhealthy, size=resize[::-1],mode='trilinear',align_corners=True)	
			y_lesion = batch_no_channel_interp3d(y_lesion,tuple(resize[::-1]),mode='nearest')
			y_healthy = batch_no_channel_interp3d(y_healthy,tuple(resize[::-1]),mode='nearest')
		return x_healthy, x_unhealthy, y_lesion, y_healthy

	def generate_multi_channel_data(self, channel_size=6,random_mode=True):
		x1, xu = [],[]
		for j in range(channel_size):
			x_healthy, x_unhealthy, y_lesion = self.generate_data(random_mode=random_mode)
			y_healthy = y_lesion*0
			x1.append(x_healthy)
			xu.append(x_unhealthy)
			if j == 0:
				yu = y_lesion
				y1 = y_healthy
		x1 = np.stack(x1)
		xu = np.stack(xu)

		return x1, xu, yu, y1
		