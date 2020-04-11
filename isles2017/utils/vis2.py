from utils.utils import *

class SlidingVisualizer2(object):
	def __init__(self, do_show=True):
		super(SlidingVisualizer2, self).__init__()
		self.canonical_modalities_label = ['ADC','MTT','rCBF','rCBV' ,'Tmax','TTP','OT']
		self.canonical_modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP',6:'OT'}
		self.do_show = do_show
		self.use_overlay = True

	def vis1(self, one_case):
		BRIGHTNESS_FACTOR = 2
		z_index = 0
		modality_index = 0
		current_x = one_case['imgobj'][self.canonical_modalities_dict[modality_index]]
		current_s = current_x.shape

		fig, ax = plt.subplots()
		this_cmap = 'inferno'

		#################
		if self.use_overlay:
			overlay = one_case['imgobj']['pred'][:,:,:]*BRIGHTNESS_FACTOR # Use this to overlay the lesion on other modalities
			print('overlay.shape:%s'%(str(overlay.shape)))
			current_x = current_x*(1- overlay ) + overlay
		#################

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
		s_mod = Slider(ax_mod, 'modality', 0, 7, valinit=modality_index, valstep=1)
		ax_cbar = plt.axes([posx, posy + 0.05*2, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_cbar = Slider(ax_cbar, 'cbar factor', 0, 1, valinit=1, valstep=0.01)
		rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
		radio = RadioButtons(rax, ('overlay','None'), active=0)
		
		def colorfunc(label):
			self.use_overlay = False
			if label == 'overlay':
				self.use_overlay = True
			update(None)
		radio.on_clicked(colorfunc)

		def update(val):
			current_x = one_case['imgobj'][self.canonical_modalities_dict[s_mod.val]]
			#################
			if self.use_overlay:
				this_modality = self.canonical_modalities_dict[s_mod.val]
				if not (this_modality == 'OT' or this_modality=='pred') :
					overlay = one_case['imgobj']['pred'][:,:,:]*BRIGHTNESS_FACTOR # Use this to overlay the lesion on other modalities
					current_x = current_x*(1- overlay ) + overlay
			################
			current_s = current_x.shape
			ax.set_title(self.canonical_modalities_dict[s_mod.val])
			l.set_data(current_x[:,:,int(s_z_index.val)])
			l.set_clim(vmin = np.min(current_x), vmax = s_cbar.val * np.max(current_x))
			fig.canvas.draw_idle()

		s_z_index.on_changed(update)
		s_mod.on_changed(update)
		s_cbar.on_changed(update)
		if self.do_show: plt.show()