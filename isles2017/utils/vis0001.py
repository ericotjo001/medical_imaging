import utils.vis as vis
from utils.utils import *

class SlidingVisualizerMultiPanelLRP(vis.SlidingVisualizer):
	"""docstring for SlidingVisualizerMultiPanelLRP"""
	def __init__(self,):
		super(SlidingVisualizerMultiPanelLRP, self).__init__()

	def vis_filter_sweeper(self, config_data, lrp_filter_sweep_dictionary, x1, case_number, verbose=100):
		print("vis_filter_sweeper()")
		if verbose>=100:
			for xkey in lrp_filter_sweep_dictionary:
				print("  %60s | %-20s"%(str(xkey), str(lrp_filter_sweep_dictionary[xkey].shape)))
	
		dice_scores_collection = vis.load_dice_scores(config_data)
		this_dice = dice_scores_collection['dice_scores1']['latest'][case_number]

		lrp_filter_sweep_dictionary[('x')] = x1 
		x0 = lrp_filter_sweep_dictionary[('raw')]
		s = x0.shape
		depth = s[3]

		from utils.vis_header import rewrap_filter_sweep_dictionary 
		fraction_pass_filter_label, fraction_pass_filter_list = rewrap_filter_sweep_dictionary('fraction_pass_filter', lrp_filter_sweep_dictionary)
		fraction_clamp_filter_label, fraction_clamp_filter_list = rewrap_filter_sweep_dictionary('fraction_clamp_filter', lrp_filter_sweep_dictionary)

		fig = plt.figure()
		ax = fig.add_subplot(231); 
		ax2 = fig.add_subplot(232);
		ax3 = fig.add_subplot(233);
		ax4 = fig.add_subplot(234); 
		ax5 = fig.add_subplot(235); 
		ax6 = fig.add_subplot(236); 
		this_cmap = 'gray'
		axcolor = 'lightgoldenrodyellow'

		posx, posy, widthx_fraction, widthy_fraction = 0.1, .93, 0.3, 0.01
		ax_z_index = plt.axes([posx, posy, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_z_index = Slider(ax_z_index, 'z_index', 0, depth-1, valinit=0., valstep=1)
		ax_mod_index = plt.axes([posx, posy + 0.02, widthx_fraction, widthy_fraction], facecolor=axcolor)
		slider_mod_index = Slider(ax_mod_index, 'modality', 0, 6-1, valinit=0., valstep=1)
		ax_passf_index = plt.axes([posx + 0.5, posy + 0.02, widthx_fraction, widthy_fraction], facecolor=axcolor)
		slider_passf_index = Slider(ax_passf_index, 'pfilter', 0, len(fraction_pass_filter_list)-1, valinit=0., valstep=1)
		ax_clampf_index = plt.axes([posx + 0.5, posy , widthx_fraction, widthy_fraction], facecolor=axcolor)
		slider_clampf_index = Slider(ax_clampf_index, 'cfilter', 0, len(fraction_clamp_filter_list)-1, valinit=0., valstep=1)

		ax_fix_cbar = plt.axes([0.025, 0.5, 0.05, 0.075])
		ax_fix_cbar.set_title('colorbar', fontsize=10)
		radio_fix_cbar = RadioButtons(ax_fix_cbar, ('Fix','Var'), active=0)
		ax_overlay = plt.axes([0.025, 0.7, 0.05, 0.075])
		ax_overlay.set_title('overlay', fontsize=10)
		radio_overlay = RadioButtons(ax_overlay, ('Yes','No'), active=1)

		class InteractivePlot_filter_sweeper(object):
			def __init__(self,data=None):
				super(InteractivePlot_filter_sweeper, self).__init__()
				if data is not None: 
					self.data = data
				self.cb = None
				self.cb2 = None
				self.cb3 = None
				self.cb4 = None
				self.cb5 = None
				self.cb6 = None
				self.canonical_modalities_dict = {0:'ADC',1:'MTT',2:'rCBF',3:'rCBV' ,4:'Tmax',5:'TTP'}
				self.lrp_cmap = 'bwr'
				self.lrp_alpha = 1.0

			def clear_figures(self):
				self.cb1.remove()
				self.cb2.remove()
				self.cb3.remove()
				self.cb4.remove()
				self.cb5.remove()
				self.cb6.remove()
				ax.clear()
				ax2.clear()
				ax3.clear()
				ax4.clear()
				ax5.clear()
				ax6.clear()				

			def update(self, val,init=False):
				if init:
					current_mod = 0
					current_z_index = 0
					current_pfilter_index = 0
					current_cfilter_index = 0
				else:
					current_mod = int(slider_mod_index.val)
					current_z_index = int(s_z_index.val)
					current_pfilter_index = int(slider_passf_index.val)
					current_cfilter_index = int(slider_clampf_index.val)
					if radio_overlay.value_selected == 'Yes': 
						self.lrp_cmap = 'inferno'
						self.lrp_alpha = 0.7
					elif radio_overlay.value_selected == 'No':
						self.lrp_cmap = 'bwr'
						self.lrp_alpha = 1. 
					self.clear_figures()
				
				x6 = self.data[('x')][current_mod]
				maxabs6 = max(abs(x6.reshape(-1)))
				l6 = ax6.imshow(x6[:,:,current_z_index]/maxabs6, cmap=this_cmap)
				self.cb6 = plt.colorbar(l6, ax=ax6)
				ax6.set_xlabel('%s. f=%s'%(str(self.canonical_modalities_dict[current_mod]),str(maxabs6)))

				###################################
				# LRP plots
				###################################
				x = self.data[('raw')][current_mod][:,:,current_z_index]
				maxabs1 = max(abs(self.data['raw'][current_mod].reshape(-1)))
				if radio_overlay.value_selected == 'Yes':
					plt.sca(ax); plt.imshow(x6[:,:,current_z_index]/maxabs6, cmap=this_cmap)
				l = ax.imshow(x/maxabs1, cmap=self.lrp_cmap, alpha=self.lrp_alpha)
				ax.set_xlabel(self.canonical_modalities_dict[current_mod] + '(raw lrp)')
				self.cb1 = plt.colorbar(l, ax=ax)
				ax.set_title('case_number:%s. f=%s'%(str(case_number),str(maxabs1)))

				xp = fraction_pass_filter_list[current_pfilter_index][current_mod]
				maxabs2 = max(abs(xp.reshape(-1)))
				if radio_overlay.value_selected == 'Yes':
					plt.sca(ax2); plt.imshow(x6[:,:,current_z_index]/maxabs6, cmap=this_cmap)
				l2 = ax2.imshow(xp[:,:,current_z_index]/maxabs2, cmap=self.lrp_cmap, alpha=self.lrp_alpha)
				ax2.set_xlabel(str(self.canonical_modalities_dict[current_mod])+fraction_pass_filter_label[current_pfilter_index])
				self.cb2 = plt.colorbar(l2, ax=ax2)
				ax2.set_title('f=%s'%(maxabs2))

				xc = fraction_clamp_filter_list[current_cfilter_index][current_mod]
				maxabs3 = max(abs(xc.reshape(-1)))
				if radio_overlay.value_selected == 'Yes':
					plt.sca(ax3); plt.imshow(x6[:,:,current_z_index]/maxabs6, cmap=this_cmap)
				l3 = ax3.imshow(xc[:,:,current_z_index]/maxabs3, cmap=self.lrp_cmap, alpha=self.lrp_alpha)
				ax3.set_xlabel(str(self.canonical_modalities_dict[current_mod])+fraction_clamp_filter_label[current_cfilter_index])
				self.cb3 = plt.colorbar(l3, ax=ax3)
				ax3.set_title('f=%s'%(maxabs3))

				###################################
				# segmentation
				###################################
				ot = self.data[('OT')][:,:,current_z_index]
				l4 = ax4.imshow(ot, cmap=this_cmap)
				self.cb4 = plt.colorbar(l4, ax=ax4)
				ax4.set_xlabel('OT')

				y = self.data[('y')][:,:,current_z_index]
				l5 = ax5.imshow(y, cmap=this_cmap)
				self.cb5 = plt.colorbar(l5, ax=ax5)
				ax5.set_xlabel('y. dice=%s'%(str(this_dice)))



				if radio_fix_cbar.value_selected =='Fix':
					l.set_clim(vmin=-1,vmax=1)
					l2.set_clim(vmin=-1,vmax=1)
					l3.set_clim(vmin=-1,vmax=1)
					l6.set_clim(vmin=0,vmax=1)
				l4.set_clim(vmin=0,vmax=1)
				l5.set_clim(vmin=0,vmax=1)
				if not init: fig.canvas.draw_idle()
		
		ip = InteractivePlot_filter_sweeper(data=lrp_filter_sweep_dictionary)
		ip.update(0,init=True)
		
		s_z_index.on_changed(ip.update)
		slider_mod_index.on_changed(ip.update)
		slider_passf_index.on_changed(ip.update)
		slider_clampf_index.on_changed(ip.update)
		radio_fix_cbar.on_clicked(ip.update)
		radio_overlay.on_clicked(ip.update)

		plt.show()
		plt.tight_layout()	