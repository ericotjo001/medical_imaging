from utils.debug_switches import *
from utils.packages import *
from utils.routine_functions import *
from utils.description1 import *
from utils.description2 import *
from utils.config_data_default import *

this_device = torch.device('cuda:0')
number_of_classes = 2 # 0,1

 
def interp3d(x,size, mode='nearest'):
	'''
	x is a 3D tensor

	# x = np.random.randint(0,10,size=(2,3,4))
	# x = torch.Tensor(x)
	# z = F.interpolate(x, size=(8), mode='nearest')
	# print(x.shape)
	# print(z)
	# print(interp3d(x,(2,3,4)))
	'''
	x = F.interpolate(x, size=(int(size[2])), mode=mode)
	x = x.transpose_(1,2)
	x = F.interpolate(x, size=(int(size[1])), mode=mode)
	x = x.transpose_(1,2)
	x = x.transpose_(0,2)
	x = F.interpolate(x, size=(int(size[0])), mode=mode)
	x = x.transpose_(0,2)
	return x

def batch_interp3d(x,size,mode='nearest'):
	'''
	x is of the form 5D, (batch_size,C,D,H,W)
	'''
	s = tuple(x.shape)
	out = torch.zeros((s[0],s[1])+ size)

	for i in range(s[0]):
		for j in range(s[1]):
			out[i,j,:,:,:] = interp3d(x[i,j,:,:,:],size, mode=mode)
	return out

def batch_no_channel_interp3d(x,size,mode='nearest'):
	'''
	x is of the form 4D, (batch_size,D,H,W)
	'''
	s = tuple(x.shape)
	out = torch.zeros((s[0],)+ size)

	for i in range(s[0]):
		out[i,:,:,:] = interp3d(x[i,:,:,:],size, mode=mode)
	return out

def get_zero_container(x,y):
	"""
	Assume x, y are tensors of the same dimension
	but not necessarily have the same size
	for example x can be 3,4,5 and y 4,3,5
	return the size that can contain both: 4,4,5
	
	Example:
	x = torch.tensor(np.random.normal(0,1,size=(3,4,5)))
	y = torch.tensor(np.random.normal(0,1,size=(4,3,5)))
	z = get_zero_container(x,y)
	print(z.shape) # torch.Size([4, 4, 5])
	"""
	s = []
	for sx,sy in zip(x.shape,y.shape):
		s.append(np.max([sx,sy]))
	return torch.zeros(s)


def centre_crop_tensor(x, intended_shape):
	"""
	Assume x is pytorch tensor
	Given tensor x of shape (4,6), if we want to extract
	  its center of size (4,4), then set intended_shape=(4,4)
	Any dimension of x works.
	Example:
		x = torch.tensor(np.random.randint(0,10,size=(4,6)))
		x1 = centre_crop_tensor(x, (4,4))
		print(x)
		print(x1)
	Example 2:
	for i in range(1000):
		randomsize = np.random.randint(50,100,size=(1,2))[0]
		randomcropsize = np.random.randint(10,48,size=(1,2))[0]
		x = torch.tensor(np.random.randint(0,10,size=randomsize))
		x1 = centre_crop_tensor(x, randomcropsize)
		print("%s|| %s == %s"%(str(randomsize),str(randomcropsize),str(np.array(x1.shape))))

	"""
	limits = []
	for s,s1 in zip(x.shape,intended_shape):
		if s == s1: 
			limits.append(slice(0,s,None))
		else:
			diff1 = np.floor(np.abs(s-s1)/2)
			diff2 = np.ceil(np.abs(s-s1)/2)
			limits.append(slice(int(diff1),int(s-diff2),None))
	limits = tuple(limits)
	x1 = x[limits]
	# print(np.array(x1.shape),intended_shape)
	assert(np.all(np.array(x1.shape)==intended_shape))
	return x1



class SaveableObject(object):
	def __init__(self, ):
		super(SaveableObject, self).__init__()
		self.a = None

	def set_a(self, a):
		self.a = a

	def save_object(self,fullpath):
		output = open(fullpath, 'wb')
		pickle.dump(self, output)
		output.close()

	def load_object(self, fullpath):
		pkl_file = open(fullpath, 'rb')
		this_loaded_object = pickle.load(pkl_file)
		pkl_file.close() 
		return this_loaded_object

def intersect_fraction_a_in_b(a,b):
	'''
	a, b binary np.array of the same size with dtype float
	Motivation: 
	  if b is proper subset of a, then output is 1
	  if a is proper subset of b, then sum(b intersect a)/sum(b) 
	b is the limiting factor

	a = np.random.randint(0,2,(4,4)).astype(np.float)
	b = np.random.randint(0,2,(4,4)).astype(np.float)

	m = intersect_fraction_a_in_b(a,b)
	print(a, sum(a.reshape(-1)))
	print(b, sum(b.reshape(-1)))
	print()
	print(a*b)
	print(m)
	'''
	y = a*b
	return sum(y.reshape(-1))/sum(b.reshape(-1))