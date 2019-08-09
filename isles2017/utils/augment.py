from utils.utils import *

class Image3DRotator(object):
	def __init__(self):
		super(Image3DRotator, self).__init__()
		
	def rotate_img3D(self, x, theta_deg, ax=0):
		'''
		Assume x is in the shape d,h,w
		'''
		s = x.shape
		y = np.zeros(s)
		for i in range(s[ax]):
			if ax == 0: y[i,:,:] = self.rotate_img(x[i,:,:],theta_deg)
			if ax == 1: y[:,i,:] = self.rotate_img(x[:,i,:],theta_deg)
			if ax == 2: y[:,:,i] = self.rotate_img(x[:,:,i],theta_deg)
		return y
	def rotate_img(self, x,theta_deg):
		'''Assume x is numpy array 2D of shape (w,h)'''
		return np.array(Image.fromarray(x).rotate(theta_deg))

class Image3DClipTranslate(object):
	def __init__(self):
		super(Image3DClipTranslate, self).__init__()

	def clip_and_translate(self, x, pixel=1, clip_min_max = [0,1], ax=0):
		'''
		Assume x is in the shape d,h,w
		'''
		x = np.clip(x,clip_min_max[0],clip_min_max[1])
		s = x.shape
		y = np.zeros(s)
		if pixel > 0:
			if ax == 0: y[pixel:,:,:] = x[:-pixel,:,:]
			elif ax == 1: y[:,pixel:,:] = x[:,:-pixel,:]
			elif ax == 2: y[:,:,pixel:] = x[:,:,:-pixel]
		elif pixel<0:
			pixel = int(np.abs(pixel))
			if ax == 0: y[:-pixel,:,:] = x[pixel:,:,:]
			elif ax == 1: y[:,:-pixel,:] = x[:,pixel:,:]
			elif ax == 2: y[:,:,:-pixel] = x[:,:,pixel:]
		else:
			y = x
		return y 

class Visualizer3D(Image3DRotator, Image3DClipTranslate):
	"""docstring for Visualizer3D"""
	def __init__(self,):
		super(Visualizer3D, self).__init__()
		self.do_show = True

	def vis1(self, x):
		z_index = 0
		current_x = x.copy() 
		current_s = current_x.shape

		fig,ax= plt.subplots()
		plt.subplots_adjust(left=0.25, bottom=0.25)
		l = plt.imshow(current_x[:,:,z_index],cmap='gray')
		l.set_clim(vmin = np.min(current_x), vmax = np.max(current_x))
		ax.invert_yaxis()
		plt.colorbar()
		'''
		Sliders
		'''
		axcolor = 'lightgoldenrodyellow'
		posx, posy, widthx_fraction, widthy_fraction = 0.25, 0.1, 0.65, 0.03 
		ax_z_index = plt.axes([posx, posy, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_z_index = Slider(ax_z_index, 'z_index', 0, current_s[2] - 1, valinit=z_index, valstep=1)
		ax_cbar = plt.axes([posx, posy + 0.05*2, widthx_fraction, widthy_fraction], facecolor=axcolor)
		s_cbar = Slider(ax_cbar, 'cbar factor', 0, 1, valinit=1, valstep=0.05)

		def update(val):
			current_x = x.copy()
			current_s = current_x.shape
			l.set_data(current_x[:,:,int(s_z_index.val)])
			l.set_clim(vmin = np.min(current_x), vmax = s_cbar.val * np.max(current_x))
			fig.canvas.draw_idle()
		s_z_index.on_changed(update)
		s_cbar.on_changed(update)
		if self.do_show: plt.show()

	def vis2(self, x, x_mod):
		self.do_show = True
		p = Pool(2)
		p.map(self.vis1, [x, x_mod])