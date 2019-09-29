from utils.utils import *


class SlidingVisualizer(object):
	"""docstring for SlidingVisualizer"""
	def __init__(self):
		super(SlidingVisualizer, self).__init__()
		self.canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
		self.canonical_modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP',6:'OT'}
		self.do_show = False

	def vis1(self, one_case):
		z_index = 0
		modality_index = 0
		current_x = one_case['imgobj'][self.canonical_modalities_dict[modality_index]]
		current_s = current_x.shape

		fig,ax= plt.subplots()
		plt.subplots_adjust(left=0.25, bottom=0.25)
		l = plt.imshow(current_x[:,:,z_index],cmap='gray')
		l.set_clim(vmin = np.min(current_x), vmax = np.max(current_x))
		ax.set_title(self.canonical_modalities_dict[modality_index])

		plt.colorbar()
		'''
		Sliders
		'''
		axcolor = 'lightgoldenrodyellow'
		posx, posy, widthx_fraction, widthy_fraction = 0.25, 0.1, 0.65, 0.03 
		ax_z_index = plt.axes([posx, posy, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_z_index = Slider(ax_z_index, 'z_index', 0, current_s[2] - 1, valinit=z_index, valstep=1)
		ax_mod = plt.axes([posx, posy + 0.05, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_mod = Slider(ax_mod, 'modality', 0, 6, valinit=modality_index, valstep=1)
		ax_cbar = plt.axes([posx, posy + 0.05*2, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_cbar = Slider(ax_cbar, 'cbar factor', 0, 1, valinit=1, valstep=0.01)

		def update(val):
			current_x = one_case['imgobj'][self.canonical_modalities_dict[s_mod.val]]
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
		lrp = ax.imshow(current_Rc[:,:,z_index],cmap='bwr',alpha=0.5) #
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
			lrp.set_clim(vmin= -lim,vmax=lim)
			
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
		lrp.set_clim(vmin= np.min(Rc),vmax=np.max(Rc))
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
			fig.canvas.draw_idle()

		s_z_index.on_changed(update)
		s_mod.on_changed(update)
		s_cbar.on_changed(update)
		if self.do_show: plt.show()