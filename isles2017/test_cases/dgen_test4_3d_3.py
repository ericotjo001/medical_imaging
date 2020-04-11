import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *

from dataio.data_diffgen import DG3D
dg = DG3D(unit_size=(192,192))
x1, xu, yu, y1= dg.generate_multi_channel_data(channel_size=6)
print(x1.shape)
print(xu.shape)
print(yu.shape)
print(y1.shape)