from multivh3 import *
from multivtests_aux import *

'''
2D TEST
'''
def multiv_uniform2D_test(config_dir=None, verbose=10):
	if verbose > 9: print("Running multiv_uniform2D_test().")
	x, reconstructed = multiv_uniform('2', config_dir, verbose=verbose)
	return x, reconstructed

def multiv_unirand2D_test(config_dir=None, verbose=10):
	if verbose > 9: print("Running multiv_unirand2D_test().")
	multiv_unirand('2', config_dir, verbose=verbose)
	return

def dualview2D_test(config_dir=None, verbose=10):
	if verbose > 9: print("Running dualview2D_test().")
	dualview('2', config_dir, verbose=10)

def dualuniform2D_test(config_dir=None, verbose=10):
	if verbose > 9: print("Running dualuniform2D_test().")
	dualuniform('2', config_dir, verbose=20)

'''
3D TEST
'''
def multiv_uniform3D_test(config_dir=None, verbose=10): 
	if verbose > 9: print("Running multiv_uniform3D_test().")
	x, reconstructed = multiv_uniform('3', config_dir, verbose=verbose)
	return x, reconstructed

def multiv_unirand3D_test(config_dir=None, verbose=10):
	if verbose > 9: print("Running multiv_unirand3D_test().")
	multiv_unirand('3', config_dir, verbose=verbose)
	return

def dualview3D_test(config_dir=None, verbose=10):
	if verbose > 9: print("Running dualview3D_test().")
	dualview('3', config_dir, verbose=20)

def dualuniform3D_test(config_dir=None, verbose=10):
	if verbose > 9: print("Running dualuniform2D_test().")
	dualuniform('3', config_dir, verbose=20)

'''
DETAILED TESTS IMPLEMENTATIONS
'''
def dualuniform(dim, config_dir, verbose=10):
	if dim == '2': config_keyword = 'dualuniform2D_test'
	elif dim == '3': config_keyword = 'dualuniform3D_test'
	else: raise Exception('dimension error!')
	if config_dir is None: config_dir = 'multivconfig.json'
	conf = ConfigParser(config_dir, ['test', config_keyword])
	data_dict = conf.get_config_data(verbose=10)

	""" Generate random testing data """
	x, img_arr = aux00002_generate_data(dim, data_dict, verbose)

	if dim == '2': 
		mN = MultiViewSlicer2D(x.shape)
		mN.multiview_slice_and_dice_zero_padded(x, data_dict['slice_shape'], data_dict['shell_extension'], verbose)
		Img_arr = mN.X.transpose()
	elif dim == '3':
		mN = MultiViewSlicer3D(x.shape)
		mN.multiview_slice_and_dice_zero_padded(x, data_dict['slice_shape'], data_dict['shell_extension'], verbose)
		Img_arr = mN.X

	filenames = []	
	gifname = config_keyword + '.gif'
	'''
	Note that minor and major slices are slices of mN.X, i.e. the padded version, not x
	'''
	for i, the_slices in enumerate(zip(mN.slice_collection , mN.minor_view_slice_collection , mN.major_view_slice_collection)):
		""" Get the rectangles marking the slices on the exploded view """
		this_slice, minor_slice, major_slice = tuple(the_slices[0]), tuple(the_slices[1]), tuple(the_slices[2])
		if dim == '2':
			temp_dir, fps = 'gif_making_dual_temp', 1.2
			xran, yran = range(mN.X.shape[0])[minor_slice[0]], range(mN.X.shape[1])[minor_slice[1]]
			xSran, ySran = range(mN.X.shape[0])[major_slice[0]], range(mN.X.shape[1])[major_slice[1]]
			rect = patches.Rectangle((xran[0], yran[0]),len(xran),len(yran),linewidth=3,edgecolor='g',facecolor='none')	
			rectS = patches.Rectangle((xSran[0], ySran[0]),len(xSran),len(ySran),linewidth=5,edgecolor='r',facecolor='none')			
			delineate, delineate_shell = rect, rectS
			filename = aux00003_3_generate_one_static_image_frame(dim, Img_arr, i, delineate, delineate_shell, temp_dir)
		if dim=='3':
			temp_dir, fps = 'gif3D_making_dual_temp', 0.4
			""" Get the cube marking the slice"""
			from mpl_toolkits.mplot3d import Axes3D
			filename = aux00003_3_generate_one_static_image_frame(dim, Img_arr, i, minor_slice, major_slice, temp_dir)
		filenames.append(filename)
	if data_dict['save_gif']: aux00001_1_save_gif_from_temp_folder(filenames, gifname, dim, fps=fps)	
	if not data_dict['save_static_figures']: print("\nRemoving temp img folder...") ;shutil.rmtree(temp_dir)

def dualview(dim, config_dir, verbose=10):
	if dim == '2': config_keyword = 'dualview2D_test'
	elif dim == '3': config_keyword = 'dualview3D_test'
	else: raise Exception('dimension error!') 

	if config_dir is None: config_dir = 'multivconfig.json'
	conf = ConfigParser(config_dir, ['test', config_keyword])
	data_dict = conf.get_config_data(verbose=10)

	""" Generate random testing data """
	x, img_arr = aux00002_generate_data(dim, data_dict, verbose)

	""" Preparing multiview sampler objects """
	if verbose > 9: print("\nInitiating MultiViewSampler" + str(dim) + "D object.")
	if dim == '2': sN = MultiViewSampler2D(data_dict['shapeND'], data_dict['slice_shape'], data_dict['shell_shape'])
	elif dim == '3': sN = MultiViewSampler3D(data_dict['shapeND'], data_dict['slice_shape'], data_dict['shell_shape'])	
	sN.set_center_range(padding=None, verbose=10)

	""" Testing the slices validity """
	if verbose > 9: print("\nBegin range testing...",end='')
	check_result = sN.test_range(n_tries=1000, verbose=0)
	assert(check_result)
	if verbose > 9: print("Passed the test.")
	
	filenames = []	
	gifname = config_keyword + '.gif'
	for i in range(data_dict['n_sample']):
		slice_center = sN.get_center()
		one_slice = sN.get_slice(slice_center)
		one_shell_slice = sN.get_shell_slice(slice_center)
		if dim=='2':
			temp_dir, fps = 'gif_making_dual_temp', 1.2
			""" Get the rectangles marking the slices"""
			xran, yran = range(sN.full_shape[0])[one_slice[0]], range(sN.full_shape[1])[one_slice[1]]
			xSran, ySran = range(sN.full_shape[0])[one_shell_slice[0]], range(sN.full_shape[1])[one_shell_slice[1]]
			rect = patches.Rectangle((xran[0], yran[0]),len(xran),len(yran),linewidth=3,edgecolor='g',facecolor='none')	
			rectS = patches.Rectangle((xSran[0], ySran[0]),len(xSran),len(ySran),linewidth=5,edgecolor='r',facecolor='none')			
			delineate, delineate_shell = rect, rectS
			""" Create the Image """
			filename = aux00003_2_generate_one_static_image_frame(dim, img_arr, i, slice_center, 
				one_slice, delineate, one_shell_slice, delineate_shell, temp_dir)
		if dim=='3':
			temp_dir, fps = 'gif3D_making_dual_temp', 0.4
			""" Get the cube marking the slice"""
			from mpl_toolkits.mplot3d import Axes3D
			filename = aux00003_2_generate_one_static_image_frame(dim, img_arr, i, slice_center, 
				one_slice, None, one_shell_slice, None, temp_dir)

		filenames.append(filename)

	if data_dict['save_gif']: aux00001_1_save_gif_from_temp_folder(filenames, gifname, dim, fps=fps)	
	if not data_dict['save_static_figures']: print("\nRemoving temp img folder...") ;shutil.rmtree(temp_dir)

def multiv_unirand(dim, this_dir, verbose=10):
	if dim=='2': config_keyword = 'multiv_unirand2D_test'	
	elif dim=='3': config_keyword = 'multiv_unirand3D_test'
	else: raise Exception('dimension error!') 

	if this_dir is None: this_dir = 'multivconfig.json'
	conf = ConfigParser(this_dir, ['test', config_keyword])
	data_dict = conf.get_config_data(verbose=10)
	
	""" Generate random testing data """
	x, img_arr = aux00002_generate_data(dim, data_dict, verbose=20)

	""" Preparing sampler objects """
	if verbose > 9: print("\nInitiating Sampler" + str(dim) + "D object.")
	if dim == '2': sN = Sampler2D(data_dict['shapeND'], data_dict['slice_shape'])
	elif dim == '3': sN = Sampler3D(data_dict['shapeND'], data_dict['slice_shape'])
	sN.set_center_range(padding=None, verbose=10)
		
	""" Testing the slices validity """
	if verbose > 9: print("\nBegin range testing...",end='')
	check_result = sN.test_range(n_tries=1000, verbose=0)
	assert(check_result)
	if verbose > 9: print("Passed the test.")

	filenames = []			
	for i in range(data_dict['n_sample']):
		slice_center = sN.get_center()
		one_slice = sN.get_slice(slice_center)
		if dim=='2':
			temp_dir, fps = 'gif_making_temp', 1.5
			""" Get the rectangle marking the slice"""
			xran, yran = range(sN.full_shape[0])[one_slice[0]], range(sN.full_shape[1])[one_slice[1]]
			rect = patches.Rectangle((xran[0], yran[0]),len(xran),len(yran),linewidth=5,edgecolor='r',facecolor='none')			
			filename = aux00003_generate_one_static_image_frame(dim=dim, img_arr=img_arr, i=i, 
				slice_center=slice_center, one_slice=one_slice, delineate=rect, temp_dir=temp_dir)
		if dim=='3':
			temp_dir, fps = 'gif3D_making_temp', 0.4
			""" Get the cube marking the slice"""
			from mpl_toolkits.mplot3d import Axes3D
			filename = aux00003_generate_one_static_image_frame(dim=dim, img_arr=img_arr, i=i, 
				slice_center=slice_center, one_slice=one_slice, delineate=None, temp_dir=temp_dir)
		filenames.append(filename)
	if data_dict['save_gif']: aux00001_save_gif_from_temp_folder(filenames, dim, fps=fps)	
	if not data_dict['save_static_figures']: print("\nRemoving temp img folder...") ;shutil.rmtree(temp_dir)

	return

def multiv_uniform(dim, this_dir, verbose=10):
	if dim=='2': config_keyword = 'multiv_uniform2D_test'	
	elif dim=='3': config_keyword = 'multiv_uniform3D_test'
	else: raise Exception('dimension error!') 

	if this_dir is None: this_dir = 'multivconfig.json'
	conf = ConfigParser(this_dir, ['test', config_keyword])
	shapeND, slice_shape = conf.get_config_data(verbose=10)

	""" Generate random testing data """
	x = np.random.randint(0,10,size=shapeND)
	if verbose > 9:	print("\nArrays:\n",x)
	if verbose > 9: print("\nInitiating slicer object Slice%sD()."%(str(dim)))

	""" Preparing slicer objects """
	if dim=='2': sN = Slice2D(list(x.shape))	
	elif dim=='3': sN = Slice3D(list(x.shape))	
	sN.set_slice_shape(slice_shape)
	if verbose > 9: sN.print_general_info()

	""" Display the uniform slices """
	sN.get_uniform_slices_no_padding()
	reconstruction_mapping = []
	if verbose > 9: print("\nGetting uniform slices, no padding:")
	for this_slice in sN.slice_collection: 
		if verbose > 9: print(this_slice,"\n", x[this_slice])
		reconstruction_mapping.append((this_slice, x[this_slice])) 

	""" Show that the slices can be reconstructed to the original image """
	if verbose > 9: print("\nBegin reconstruction")
	reconstructed = sN.reconstruct_uniform_slices_no_padding(reconstruction_mapping) # .astype(np.int)
	check = np.all(reconstructed == x)
	if verbose > 9: print("Reconstruction outcome:", check, "\n", reconstructed)
	return x, reconstructed

def multiv_test1():
	print("Running multiv_test1().")
	print("  Note: set --submode to run other tests.")
	return
