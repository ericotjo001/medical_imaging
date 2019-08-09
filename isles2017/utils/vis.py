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
		s_cbar = Slider(ax_cbar, 'cbar factor', 0, 1, valinit=1, valstep=0.05)

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