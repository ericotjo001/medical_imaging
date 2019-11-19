import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

W,H,D = 8,8,20
data = np.ones(shape=(W,H,D))
for k in range(D): data[:,:,k] = data[:,:,k] + np.random.uniform(0,1, size=(W,H))
for j in range(H): data[:,j,:] = data[:,j,:] + 1*j
data[0:40,0:40,0] = -10
data[0:3,0:3,5] = 30

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax2.hist(data.reshape(-1))
ax3 = fig.add_subplot(223)
axcolor = 'lightgoldenrodyellow'
posx, posy, widthx_fraction, widthy_fraction = 0.1, .93, 0.3, 0.01

ax_depth = plt.axes([posx, posy, widthx_fraction, widthy_fraction], facecolor=axcolor)
slider_depth = Slider(ax_depth, 'Depth', 0, D-1, valinit=0, valstep=1)

rax = plt.axes([0.025, 0.5, 0.05, 0.075])
rax.set_title('colorbar')
radio = RadioButtons(rax, ('yes','nope'), active=0)

class InteractivePlot(object):
	def __init__(self):
		super(InteractivePlot, self).__init__()
		self.cb = None
		self.cb3 = None
		self.show_cbar = True

	def update(self, val, init=False):
		if init:
			current_depth = 0
		else:
			current_depth = int(slider_depth.val)
			if self.show_cbar:
				self.cb1.remove()
				self.cb3.remove()
		
		x = data[:,:,current_depth]
		l = ax.imshow(x)
		self.cb1 = plt.colorbar(l, ax=ax)
		
		l3 = ax3.imshow(x)
		l3.set_clim(vmin=min(x.reshape(-1)),vmax=max(x.reshape(-1)))
		self.cb3 = plt.colorbar(l3, ax=ax3)
			
		l.set_clim(vmin=min(data.reshape(-1)),vmax=max(data.reshape(-1)))
		if not init: fig.canvas.draw_idle()
		
		if radio.value_selected=='nope':
			self.show_cbar = False
			self.cb1.remove()
			self.cb3.remove()
		else: 
			self.show_cbar = True	


ip = InteractivePlot()	
ip.update(0, init=True)
slider_depth.on_changed(ip.update)
radio.on_clicked(ip.update)

plt.show()
