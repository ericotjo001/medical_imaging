import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

from dataio.data_diffgen import DGen

dg = DGen(unit_size=(64,64))
dim = len(dg.unit_size)

init_coord = dg.get_init_coord(dg.unit_size, init_point='center')
# init_coord=np.array([5,5])
directions=(1,1)
n_steps=21
seq = dg.grow_sequences_of_coordinates_in_singular_direction(init_coord, directions, n_steps)
enlarge_range_forward, enlarge_range_backward = [9,4], [5,3]
new_set_of_coords = dg.enlarge_sequence(seq,enlarge_range_forward, enlarge_range_backward,
	param=0.01, lower_threshold=0.1, block_factor=0.5)
x = dg.get_binary_map_from_coords(new_set_of_coords, dim)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(x)
plt.show()
# x = dg.generate_one_tree(min_val=0.0, max_val=1.0, init_point='center', init_value=1.0)