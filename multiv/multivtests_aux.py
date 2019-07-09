from multivh3 import *


def aux00001_save_gif_from_temp_folder(filenames, dim, fps=1.5):
	gifname = 'multiv_unirand'+str(dim)+'D.gif'
	aux00001_1_save_gif_from_temp_folder(filenames, gifname, dim, fps=fps)
	return

def aux00001_1_save_gif_from_temp_folder(filenames, gifname, dim, fps=1.5):
	try: import imageio
	except: raise Exception("Error at import imageio. Do pip install imageio if you still want to save gif image.")
	print("\nSaving gif...", end='')
	images = []
	for filename in filenames:
		images.append(imageio.imread(filename))
	imageio.mimsave(gifname, images, fps=fps)
	print("  Saved as " + gifname)
	return

def aux00002_generate_data(dim, data_dict, verbose):
	x, img_arr = None, None	
	if dim=='2':
		x = np.random.randint(0, 10, size=data_dict['shapeND'])
		img_arr = x.transpose()
		if verbose > 9: print("\ntransposed array:\n",np.flipud(img_arr))
		return x, img_arr
	if dim=='3':
		x = np.random.randint(0, 10, size=data_dict['shapeND'])
		img_arr = x
		if verbose > 9: print("\ntransposed(2,1,0) array:\n",x.transpose(2,1,0))
		if verbose > 19: aux00002_1(x) # to plot the volume and check the array's values in the slice.
		return x, img_arr
	return

def aux00002_1(x):
	from mpl_toolkits.mplot3d import Axes3D
	alpha=0.2
	img_arr = x
	s = img_arr.shape
	filled, filled2 = np.ones(s), np.zeros(s)
	filled2[:,:,1] = 1
	colors = np.zeros(s + (4,))
	colors[:,:,:,2], colors[:,:,:,3] = img_arr/np.max(img_arr), alpha

	fig = plt.figure(figsize=(12,6))
	plt.title("checking the array values of a slice")
	plt.axis('off')
	ax = fig.add_subplot(121,projection='3d')
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	ax.voxels(filled, facecolors=colors, edgecolors='grey')
	ax.voxels(filled2, facecolors="red", edgecolors='grey')
	ax2 = fig.add_subplot(122, projection='3d')
	ax2.set_xlabel("x")
	ax2.set_ylabel("y")
	ax2.set_zlabel("z")
	ax2.voxels(filled2, facecolors="#ff000030", edgecolors='grey')	
	Xiter=np.nditer(filled2,flags=['f_index','multi_index'])
	while not Xiter.finished:
		temp = Xiter.multi_index
		if Xiter[0]:
			ax2.text(temp[0],temp[1],temp[2]+2, img_arr[Xiter.multi_index], c='b', fontsize=12)
		Xiter.iternext()
	plt.tight_layout()

	plt.show()
	return

def aux00003_generate_one_static_image_frame(dim, img_arr, i, slice_center, 
	one_slice, delineate, temp_dir):
	
	if not os.path.exists(temp_dir): os.mkdir(temp_dir)
	if dim == '2':
		rect = delineate
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.pcolor(img_arr, cmap='Greys')
		ax.scatter(slice_center[0]+0.5,slice_center[1]+0.5, c='r', marker='x')
		ax.add_patch(rect)
		ax.set_title( str(i) + ":" + str(one_slice))
		plt.colorbar(cax) 
	if dim == '3':
		alpha=0.2
		s = img_arr.shape
		filled, filled2, filled3 = np.ones(s), np.zeros(s), np.zeros(s)
		filled2[one_slice], filled3[slice_center] = 1, 1
		xran, yran, zran = range(s[0])[one_slice[0]], range(s[1])[one_slice[1]], range(s[2])[one_slice[2]]
		xmin, xmax, ymin, ymax, zmin, zmax = np.min(xran), np.max(xran), np.min(yran), np.max(yran), np.min(zran), np.max(zran)
		
		colors = np.zeros(s + (4,))
		colors[:,:,:,2], colors[:,:,:,3] = img_arr/np.max(img_arr), alpha
		colors2, colors3 = '#ff000050', '#ff000090'  

		filled2[one_slice] = np.max(filled) + 2
		ax = make_ax(grid=True)
		ax.voxels(filled3, facecolors=colors3, edgecolors='black')
		ax.voxels(filled2, facecolors=colors2, edgecolors='grey')
		ax.voxels(filled, facecolors=colors, edgecolors=None)
		this_string = "(" + str(xmin) + "," + str(ymin) + ","  + str(zmin) + ")"
		this_string2 = "(" + str(xmax) + "," + str(ymax) + ","  + str(zmax) + ")"
		this_title = str(i) + ":" + str(one_slice)
		ax.text(xmin-0.5, ymin-0.5, zmin-0.5, s=this_string, color='y', fontsize=12)
		ax.text(xmax+0.5, ymax+0.5, zmax+1.5, s=this_string2, color='y', fontsize=12)
		ax.set_title(this_title)

	""" Save images into temp folder"""
	filename = os.path.join(temp_dir,str(i)+'.jpg')
	plt.savefig(filename)
	plt.close()
	return filename

def aux00003_2_generate_one_static_image_frame(dim, img_arr, i, slice_center, 
	one_slice, delineate, shell_slice, delineate_shell, temp_dir):
	
	if not os.path.exists(temp_dir): os.mkdir(temp_dir)
	if dim == '2':
		rect, rectS = delineate, delineate_shell
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.pcolor(img_arr, cmap='Greys')
		ax.scatter(slice_center[0]+0.5,slice_center[1]+0.5, c='r', marker='x')
		ax.add_patch(rectS)
		ax.add_patch(rect)
		ax.set_title( str(i) + ":" + str(one_slice))
		plt.colorbar(cax) 
	if dim == '3':
		alpha=0.2
		s = img_arr.shape
		filled, filled2, filled3, filled4 = np.ones(s), np.zeros(s), np.zeros(s), np.zeros(s)
		filled2[one_slice], filled3[slice_center], filled4[shell_slice] = 1, 1, 1
		xran, yran, zran = range(s[0])[shell_slice[0]], range(s[1])[shell_slice[1]], range(s[2])[shell_slice[2]]
		xmin, xmax, ymin, ymax, zmin, zmax = np.min(xran), np.max(xran), np.min(yran), np.max(yran), np.min(zran), np.max(zran)
		
		colors = np.zeros(s + (4,))
		colors[:,:,:,2], colors[:,:,:,3] = img_arr/np.max(img_arr), alpha
		colors2, colors3, colors4 = '#ff000050', '#ff000090' ,'#00550010' 

		filled2[one_slice] = np.max(filled) + 2
		ax = make_ax(grid=True)
		ax.voxels(filled4, facecolors=colors4, edgecolors='grey')
		ax.voxels(filled3, facecolors=colors3, edgecolors='black')
		ax.voxels(filled2, facecolors=colors2, edgecolors='grey')
		ax.voxels(filled, facecolors=colors, edgecolors=None)
		this_string = "(" + str(xmin) + "," + str(ymin) + ","  + str(zmin) + ")"
		this_string2 = "(" + str(xmax) + "," + str(ymax) + ","  + str(zmax) + ")"
		this_title = str(i) + ":" + str(one_slice)
		ax.text(xmin-0.5, ymin-0.5, zmin-0.5, s=this_string, color='y', fontsize=12)
		ax.text(xmax+0.5, ymax+0.5, zmax+1.5, s=this_string2, color='y', fontsize=12)
		ax.set_title(this_title)

	""" Save images into temp folder"""
	filename = os.path.join(temp_dir,str(i)+'.jpg')
	plt.savefig(filename)
	plt.close()
	return filename

def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax