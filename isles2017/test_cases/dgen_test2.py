import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

from dataio.data_diffgen import DGen

dg = DGen(unit_size=(64,64))
# x = dg.generate_one_seed_binary_tree(
# 	n_steps=10,
# 	directions=(1,1), 
# 	enlarge_range_forward=(2,2), 
# 	enlarge_range_backward=(2,2),
# 	q_constant=0.01, lower_threshold=0.1, block_factor=0.5,
# 	init_point='center')

fig = plt.figure()
fig.this_axes = []

for i, (n_steps, fig_pos) in enumerate(zip([10,20,50],[331,332,333])):
	x = dg.generate_one_seed_binary_tree(n_steps=n_steps)
	fig.this_axes.append(fig.add_subplot(fig_pos))
	fig.this_axes[-1].imshow(x)
	if i==1: fig.this_axes[-1].set_title('increasing steps')

for i, (enf, fig_pos) in enumerate(zip([(2,2),(5,5),(10,10)],[334,335,336])):
	x = dg.generate_one_seed_binary_tree(
		n_steps=30, enlarge_range_forward=enf, enlarge_range_backward=enf)
	fig.this_axes.append(fig.add_subplot(fig_pos))
	fig.this_axes[-1].imshow(x)
	if i==1: fig.this_axes[-1].set_title('increasing enlarge_ranges')

for i, (this_dir, fig_pos) in enumerate(zip([(1,1),(1,2),(4,5)],[337,338,339])):
	x = dg.generate_one_seed_binary_tree(
		n_steps=60, 
		directions=this_dir,
		enlarge_range_forward=(5,5),
		q_constant=0.01)
	fig.this_axes.append(fig.add_subplot(fig_pos))
	fig.this_axes[-1].imshow(x)
	if i==1: fig.this_axes[-1].set_title('varying direction sizes')

plt.tight_layout()

fig = plt.figure()
fig.this_axes = []
for i, (this_dir, fig_pos) in enumerate(zip([(-1,0),(1,-2),(-4,5)],[331,332,333])):
	x = dg.generate_one_seed_binary_tree(
		n_steps=60, 
		directions=this_dir,
		enlarge_range_forward=(5,5),
		q_constant=0.01)
	fig.this_axes.append(fig.add_subplot(fig_pos))
	fig.this_axes[-1].imshow(x)
	if i==1: fig.this_axes[-1].set_title('varying direction signs')
for i, (block_factor, fig_pos) in enumerate(zip([0.01,0.3,0.95],[334,335,336])):
	x = dg.generate_one_seed_binary_tree(
		n_steps=60, 
		directions=(-1,3),
		enlarge_range_forward=(10,10),
		q_constant=0.01,block_factor=block_factor)
	fig.this_axes.append(fig.add_subplot(fig_pos))
	fig.this_axes[-1].imshow(x)
	if i==1: fig.this_axes[-1].set_title('increasing block_factor')
plt.tight_layout()
plt.show()