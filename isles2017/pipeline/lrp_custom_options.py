import numpy as np
p_grid = np.array(
	[
		[0.0,0.05],
		[0.0,0.1],
		[0.0,0.2],
		[0.0,0.4],
		[0.0,0.6],
		[0.05,1.0],
		[0.1,1.0],
		[0.2,1.0],
		[0.4,1.0],
		[0.6,1.0],		
	])
p_grid_negative = np.fliplr(-p_grid)

c_grid = np.array(
	[
		[0.0,0.05],
		[0.0,0.1],
		[0.0,0.2],
		[0.0,0.4],
		[0.0,0.6],
	])
c_grid_negative = np.fliplr(-c_grid)

def get_filter_sweep_options(mode=None):
	if mode is None: 
		raise Exception('Check lrp_custom_options.py')
	elif mode=='filter_sweep_0001_OPTIONS': 
		filter_sweep_0001_OPTIONS = {
			'raw' : None,
			'fraction_pass_filter' : zip(p_grid_negative, p_grid),
			'fraction_clamp_filter': zip(c_grid_negative, c_grid)#zip(c_grid_negative, c_grid)
		}
		option_iter = filter_sweep_0001_OPTIONS
	return option_iter

if __name__=='__main__':
	this_iter = get_filter_sweep_options(mode='filter_sweep_0001_OPTIONS')['fraction_pass_filter']
	print("fraction_pass_filter")
	for xneg,xpos in this_iter:
		print(xneg,xpos) 
	
	this_iter = get_filter_sweep_options(mode='filter_sweep_0001_OPTIONS')['fraction_clamp_filter']
	print('fraction_clamp_filter')
	for xneg,xpos in this_iter:
		print(xneg,xpos) 
	"""
	[-0.3  0. ] [0.  0.3]
	[-0.5  0. ] [0.  0.5]
	[-0.7  0. ] [0.  0.7]
	[-0.9  0. ] [0.  0.9]
	[-0.5 -0.2] [0.2 0.5]
	[-0.7 -0.2] [0.2 0.7]
	[-0.9 -0.2] [0.2 0.9]
	[-0.7 -0.4] [0.4 0.7]
	[-0.9 -0.4] [0.4 0.9]
	[-0.9 -0.6] [0.6 0.9]
	"""