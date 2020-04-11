
import numpy as np


x = np.random.randint(0,10,size=(5,6,7))
s = x.shape
print(s)
print(s[::-1])