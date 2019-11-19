from utils.utils import *

def load_dice_scores(config_data):
	model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
	fullpath = os.path.join(model_dir, 'dice_scores.ev')

	pkl_file = open(fullpath, 'rb')
	dice_scores_collection = pickle.load(pkl_file)
	pkl_file.close() 
	return dice_scores_collection


class SlidingVisualizer(object):
	def __init__(self):
		super(SlidingVisualizer, self).__init__()
		self.canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
		self.canonical_modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP',6:'OT'}
		self.do_show = False

	def vis1(self, one_case, use_overlay=False):
		z_index = 0
		modality_index = 0
		current_x = one_case['imgobj'][self.canonical_modalities_dict[modality_index]]
		#################
		if use_overlay:
			overlay = one_case['imgobj']['OT']*3+1 # Use this to overlay the lesion on other modalities
			current_x = current_x *overlay
		#################
		current_s = current_x.shape

		fig, ax= plt.subplots()
		this_cmap = "gray" 
		if use_overlay: this_cmap = 'inferno'
		plt.subplots_adjust(left=0.25, bottom=0.25)
		l = plt.imshow(current_x[:,:,z_index],cmap=this_cmap)
		l.set_clim(vmin = np.min(current_x), vmax = np.max(current_x))
		ax.set_title(self.canonical_modalities_dict[modality_index])

		plt.colorbar()
		'''
		Sliders
		'''
		axcolor = 'lightgoldenrodyellow'
		posx, posy, widthx_fraction, widthy_fraction = 0.25, 0.02, 0.65, 0.03 
		ax_z_index = plt.axes([posx, posy, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_z_index = Slider(ax_z_index, 'z_index', 0, current_s[2] - 1, valinit=z_index, valstep=1)
		ax_mod = plt.axes([posx, posy + 0.05, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_mod = Slider(ax_mod, 'modality', 0, 6, valinit=modality_index, valstep=1)
		ax_cbar = plt.axes([posx, posy + 0.05*2, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_cbar = Slider(ax_cbar, 'cbar factor', 0, 1, valinit=1, valstep=0.01)


		def update(val):
			current_x = one_case['imgobj'][self.canonical_modalities_dict[s_mod.val]]
			############################
			if use_overlay:
				overlay = one_case['imgobj']['OT']*3+1 # Use this to overlay the lesion on other modalities
				current_x = current_x *overlay
			############################
			current_s = current_x.shape
			ax.set_title(self.canonical_modalities_dict[s_mod.val])
			l.set_data(current_x[:,:,int(s_z_index.val)])
			l.set_clim(vmin = np.min(current_x), vmax = s_cbar.val * np.max(current_x))
			fig.canvas.draw_idle()

		s_z_index.on_changed(update)
		s_mod.on_changed(update)
		s_cbar.on_changed(update)
		if self.do_show: plt.show()

	def vis2(self, one_case, one_case_modified):
		self.do_show = True
		p = Pool(2)
		p.map(self.vis1, [one_case, one_case_modified] )

	def vis_lrp_training(self,args):
		one_case, y, Rc, data_modalities,toggle = args
		if toggle==1:
			self.vis1_lrp_training(one_case, y, Rc, data_modalities)
		elif toggle ==2:
			self.vis2_lrp_training(one_case, y, Rc, data_modalities)
		
	def vis1_lrp_training(self, one_case, y, Rc, data_modalities):
		z_index = 0
		modality_index = 0
		current_x = one_case['imgobj'][data_modalities[modality_index]]
		current_s = current_x.shape
		current_Rc = Rc[modality_index]
		
		data_modalities_dict = {len(data_modalities):'y'}
		for i in range(len(data_modalities)): data_modalities_dict[i] = data_modalities[i]
		
		lrp_colorbar_local_limit = True
		limf = 0.5
		lim = np.max([abs(np.max(Rc)),abs(np.min(Rc))])
		
		fig,ax= plt.subplots(figsize=(9,6.5))
		plt.subplots_adjust(left=0.25, bottom=0.25)
		l = plt.imshow(current_x[:,:,z_index],cmap='gray')
		l.set_clim(vmin = np.min(current_x), vmax = np.max(current_x))
		ax.set_title(data_modalities_dict[modality_index])
		lrp = ax.imshow(current_Rc[:,:,z_index],cmap='inferno',alpha=0.5) #
		if lrp_colorbar_local_limit: lim = limf * np.max([abs(np.max(current_Rc[:,:,z_index])),abs(np.min(current_Rc[:,:,z_index]))])
		lrp.set_clim(vmin= -lim,vmax=lim)
		cb2 = plt.colorbar(lrp)
		plt.colorbar()

		'''
		Sliders
		'''
		axcolor = 'lightgoldenrodyellow'
		posx, posy, widthx_fraction, widthy_fraction = 0.25, 0.1, 0.5, 0.02
		ax_z_index = plt.axes([posx, posy, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_z_index = Slider(ax_z_index, 'z_index', 0, current_s[2] - 1, valinit=z_index, valstep=1)
		ax_mod = plt.axes([posx, posy + 0.05, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_mod = Slider(ax_mod, 'modality', 0, len(data_modalities)-1+1, valinit=modality_index, valstep=1)
		ax_cbar = plt.axes([posx, posy + 0.05*2, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_cbar = Slider(ax_cbar, 'cbar factor', 0, 1, valinit=1, valstep=0.01)

		def update(val):
			sRc = Rc.shape
			if s_mod.val<sRc[0]: # assume groundtruth OT is always arranged as the last modality
				current_Rc = Rc[int(s_mod.val)]
				current_x = one_case['imgobj'][data_modalities_dict[s_mod.val]]
			elif s_mod.val==sRc[0]: 
				current_Rc = Rc[0]*0
				current_x = one_case['imgobj'][data_modalities_dict[s_mod.val]]
			else:
				current_Rc = Rc[0]*0
				current_x = y # + np.random.normal(0,1, size=y.shape)
			current_s = current_x.shape
			
			if lrp_colorbar_local_limit: lim = limf * np.max([abs(np.max(current_Rc[:,:,int(s_z_index.val)])),abs(np.min(current_Rc[:,:,int(s_z_index.val)]))])
		
			ax.set_title(data_modalities_dict[s_mod.val])
			l.set_data(current_x[:,:,int(s_z_index.val)])
			l.set_clim(vmin = np.min(current_x), vmax = s_cbar.val * np.max(current_x))
			
			lrpdata = current_Rc[:,:,int(s_z_index.val)]
			lrp.set_data(lrpdata)
			lrp.set_clim(vmin= -s_cbar.val *lim,vmax=s_cbar.val *lim)
			
			fig.canvas.draw_idle()

		s_z_index.on_changed(update)
		s_mod.on_changed(update)
		s_cbar.on_changed(update)
		if self.do_show: plt.show()

	def vis2_lrp_training(self, one_case, y, Rc, data_modalities):
		z_index = 0
		modality_index = 0
		current_x = one_case['imgobj'][data_modalities[modality_index]]
		current_s = current_x.shape
		current_Rc = Rc[modality_index]
		
		data_modalities_dict = {len(data_modalities):'y'}
		for i in range(len(data_modalities)): data_modalities_dict[i] = data_modalities[i]
		
		lrp_colorbar_local_limit = True
		lim = np.max([abs(np.max(Rc)),abs(np.min(Rc))])
		fig,ax= plt.subplots(figsize=(9,6.5))
		plt.subplots_adjust(left=0.1, bottom=0.3)
		ax.set_title(data_modalities_dict[modality_index])
		lrp = ax.imshow(current_Rc[:,:,z_index],cmap='bwr',alpha=0.5) #
		# lrp.set_clim(vmin= np.min(Rc),vmax=np.max(Rc))
		lrp.set_clim(vmin= -lim,vmax=lim)
		cb2 = plt.colorbar(lrp)

		'''
		Sliders
		'''
		axcolor = 'lightgoldenrodyellow'
		posx, posy, widthx_fraction, widthy_fraction = 0.25, 0.1, 0.5, 0.02
		ax_z_index = plt.axes([posx, posy, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_z_index = Slider(ax_z_index, 'z_index', 0, current_s[2] - 1, valinit=z_index, valstep=1)
		ax_mod = plt.axes([posx, posy + 0.05, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_mod = Slider(ax_mod, 'modality', 0, len(data_modalities)-1+1, valinit=modality_index, valstep=1)
		ax_cbar = plt.axes([posx, posy + 0.05*2, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_cbar = Slider(ax_cbar, 'cbar factor', 0, 1, valinit=1, valstep=0.01)

		def update(val):
			sRc = Rc.shape
			if s_mod.val<sRc[0]: # assume groundtruth OT is always arranged as the last modality
				current_Rc = Rc[int(s_mod.val)]
				current_x = one_case['imgobj'][data_modalities_dict[s_mod.val]]
			elif s_mod.val==sRc[0]: 
				current_Rc = Rc[0]*0
				current_x = one_case['imgobj'][data_modalities_dict[s_mod.val]]
			else:
				current_Rc = Rc[0]*0
				current_x = y # + np.random.normal(0,1, size=y.shape)
			current_s = current_x.shape
		
			ax.set_title(data_modalities_dict[s_mod.val])
			lrpdata = current_Rc[:,:,int(s_z_index.val)]
			lrp.set_data(lrpdata)
			lrp.set_clim(vmin= -lim,vmax=lim)
			fig.canvas.draw_idle()

		s_z_index.on_changed(update)
		s_mod.on_changed(update)
		s_cbar.on_changed(update)
		if self.do_show: plt.show()


class DictionaryManager(SaveableObject):
	def __init__(self, ):
		super(DictionaryManager, self).__init__()
		self.this_dictionary = None
		self.mapping_index = {}
		self.mapping_index_inverse = {}
		"""
		self.mapping_index is a dictionary that maps the
		 keys of the main dictionary to some plottable value(s)
		"""
	def get_example(self):
		return

class DictionaryWithNumericalYArray(DictionaryManager):
	def __init__(self, ):
		super(DictionaryWithNumericalYArray, self).__init__()
		self.plainlist = {'X':[], 'Y':[]}
		self.normal_scatter_list = {'X':[], 'Y':[]}
		self.mapping_index_normal_scatter = {}
	
	def get_example(self, number_of_classes=2, size=10, verbose=10):
		XY = {}
		for i in range(number_of_classes):
			XY['x'+str(i)] = np.random.normal(0 + i,1,size=size)
		if verbose>=10:
			for x in XY:
				print("%s\n%s"%(str(x),str(XY[x])))
		return XY
	
	######################################

	def auto_get_mapping_index(self):
		for i, xkey in enumerate(self.this_dictionary):
			self.mapping_index[xkey] = i
			self.mapping_index_inverse[i] = xkey


	def get_plainlist(self, verbose=0):
		for x in self.this_dictionary:
			for this_value in self.this_dictionary[x]:
				self.plainlist['X'].append(self.mapping_index[x])
				self.plainlist['Y'].append(this_value)
		if verbose>=10:
			print("get_plainlist(): [rounded to 5 decimals]")
			for x,y, in zip(self.plainlist['X'], self.plainlist['Y']):
				print("  %s:%10s"%(str(x),str(round(y,5))))

	def scatter_plainlist(self, xlim=None, ylim=None):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(self.plainlist['X'],self.plainlist['Y'],marker='x')
		if xlim is not None: ax.set_xlim(xlim)
		if ylim is not None: ax.set_ylim(ylim)

		xticks_labels = [str(x) for x in self.this_dictionary]
		ax.set_xticks(range(len(xticks_labels)))
		ax.set_xticklabels(tuple(xticks_labels),rotation=-90)

	######################################

	def layered_boxplots(self,dict_of_XY, color_list, shift_increment=0.25, x_index_increment=1, title=None, verbose=0):
		'''

		'''
		from matplotlib.lines import Line2D

		fig = plt.figure()
		ax = fig.add_subplot(111)
		position_shift = 0.
		legend_elements = []
		legend_names = []
		for collection_name in dict_of_XY:
			if verbose>=250: print('collection_name:%s\n  %s'%(str(collection_name),str(dict_of_XY[collection_name])))
			# ax.boxplot(dict_of_XY[xkey])

			legend_elements.append(Line2D([0], [0], color=color_list[collection_name]))
			legend_names.append(collection_name)

			list_of_data = []
			xticks_labels = []
			for x in dict_of_XY[collection_name]:
				this_data = np.array(dict_of_XY[collection_name][x])
				this_data = this_data[~np.isnan(this_data)]
				list_of_data.append(this_data)
				xticks_labels.append(x)
			this_pos = np.array([1 + (i)*x_index_increment for i in range(len(list_of_data))]) + position_shift
			l = ax.boxplot(list_of_data, positions=this_pos,patch_artist=True,flierprops=dict(marker='o',markersize=1))
			for patch in l['boxes']:
				patch.set_facecolor(color_list[collection_name])
			position_shift+=shift_increment
		ax.set_xticks([1 + (i)*x_index_increment for i in range(len(list_of_data))])
		ax.set_xticklabels(tuple(xticks_labels),rotation=30)
		ax.legend(legend_elements, legend_names)
		if title is not None: plt.title(title)
		plt.tight_layout()

	######################################

	def normal_scatter_mapping_index(self):
		for i, xkey in enumerate(self.this_dictionary):
			self.mapping_index_normal_scatter[xkey] = i 

	def get_normal_scatter_list(self, mu=0,sigma=0.4, verbose=0):
		for x in self.this_dictionary:
			for this_value in self.this_dictionary[x]:
				self.normal_scatter_list['X'].append(self.mapping_index_normal_scatter[x] + np.random.normal(mu,sigma) )
				self.normal_scatter_list['Y'].append(this_value)
		if verbose>=250:
			print("get_normal_scatter_list(): [rounded to 5 decimals]")
			for x,y, in zip(self.normal_scatter_list['X'], self.normal_scatter_list['Y']):
				print("  %s:%10s"%(str(x),str(round(y,5))))
		return self.normal_scatter_list
	
	def scatter_normal_scatter_list(self, xlim=None, ylim=None):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(self.normal_scatter_list['X'],self.normal_scatter_list['Y'],marker='x')
		if xlim is not None: ax.set_xlim(xlim)
		if ylim is not None: ax.set_ylim(ylim)

		xticks_labels = [str(x) for x in self.this_dictionary]
		ax.set_xticks(range(len(xticks_labels)))
		ax.set_xticklabels(tuple(xticks_labels),rotation=30)
	
	def scatter_layered_normal_scatter_list(self, dict_of_nsXY, title=None ,xlim=None, ylim=None,marker='x',size=10):
		fig = plt.figure()
		ax = fig.add_subplot(111)

		for collection_name in dict_of_nsXY:
			nsXY = dict_of_nsXY[collection_name]
			ax.scatter(nsXY['X'],nsXY['Y'],size, marker=marker, label=collection_name)
		if xlim is not None: ax.set_xlim(xlim)
		if ylim is not None: ax.set_ylim(ylim)

		xticks_labels = [str(x) for x in self.this_dictionary]
		ax.set_xticks(range(len(xticks_labels)))
		ax.set_xticklabels(tuple(xticks_labels),rotation=30)
		ax.legend()
		if title is not None: plt.title(title)
		plt.tight_layout()
	######################################


class DictionaryWithNumericalYArrayCollection(object):
	def __init__(self, ):
		super(DictionaryWithNumericalYArrayCollection, self).__init__()
		
		