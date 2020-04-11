import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

from dataio.data_diffgen import DGen

dg = DGen(unit_size=(192,192))
dim = len(dg.unit_size)


random_translation_distance = (20,20)
assert(len(random_translation_distance)==dim)
translation_coord = np.zeros(shape=dim)
for i in range(dim):
	translation_coord[i] += np.random.randint(-random_translation_distance[i],1+random_translation_distance[i])

x = np.zeros(shape=dg.unit_size)
rand_n_seed = np.random.randint(1,12)
for i in range(rand_n_seed):
	# init_coord = dg.get_init_coord(dg.unit_size, init_point='random_center',
	# 	random_center_distance=40)
	n_steps = np.random.randint(10,20)
	this_direction = tuple([np.random.randint(-2,3),np.random.randint(-2,3)])
	x += dg.generate_one_seed_binary_tree(
		n_steps=n_steps,
		directions=this_direction, 
		enlarge_range_forward=(2,2), 
		enlarge_range_backward=(2,2),
		q_constant=0.01, lower_threshold=0.1, block_factor=0.2,
		init_point='random_center',
		init_translate=translation_coord,
		random_center_distance=10)
	x = (x>0).astype(np.float)
x = normalize_numpy_array(x,target_min=0,target_max=1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(x)
plt.show()