from utils.utils import * 
from dataio.loader_utils import *


class Cifar10Data(data.Dataset):
	"""
	import os
	from cifar2loader import *

	data_dir = os.path.join(os.getcwd(), 'data', 'cifar-10-batches-py') 
	split = 'train'
	cf = Cifar10Data(data_dir, split)
	cf2 = Cifar10Data(data_dir, 'test')

	print("cf.x_img.shape  : ", cf.x_img.shape)
	print("cf.y_label.shape: ", cf.y_label.shape)
	print("  cf.__len__() : ",cf.__len__())
	x_img1 , y_label1 = cf.__getitem__(0)
	x_img2 , y_label2 = cf.__getitem__(1)
	print("  x_img1.shape, y_label1: ",x_img1.shape, y_label1)
	print("  x_img2.shape, y_label2: ",x_img2.shape, y_label2)
	print("cf2.x_img.shape  : ", cf2.x_img.shape)
	print("cf2.y_label.shape: ", cf2.y_label.shape)
	print("  cf2.__len__() : ",cf2.__len__())

	"""
	def __init__(self):
		super(Cifar10Data, self).__init__()
		self.x_img = []	
		self.y_label = None
		self.data_size = None

		self.size_per_raw_batch = 10000
		self.img_dict = {}
		for i, name in zip(range(10),['airplane','automobile','bird', 'cat','deer','dog', 'frog','horse','ship', 'truck',]):
			self.img_dict[i] = name


	def __getitem__(self, index):
		return self.x_img[index], self.y_label[index]

	def __len__(self):
		return self.data_size

	def load_data(self, config_data, split = 'train', as_categorical=False):
		data_dir = config_data['data_directory']['cifar10']
		if config_data['data_submode']=='load_cifar10_type0001':
			self.load_cifar10_type0001(data_dir, split, as_categorical=as_categorical, num_classes=10, number_to_load=self.size_per_raw_batch)


	def load_cifar10_type0001(self, data_dir, split, as_categorical=False, num_classes=10, number_to_load=10000):
		if split == 'train':
			filename_all = []	
			for i in range(1,6):
				filename_all.append(os.path.join(data_dir, 'data_batch_'+str(i)))
			for j in range(5):
				# EACH j has 10000 data. So if number_to_load is set to 10000, there will be 50k data
			    dict1=unpickle(filename_all[j])
			    x_img_temp=dict1[b'data'].astype(np.float)
			    for i in range(number_to_load):       
			        self.x_img.append((x_img_temp[i]/255.).reshape(3,32,32))
			for i in range(5):
				dict1=unpickle(filename_all[i])
				if as_categorical:
					y_label_temp=to_categorical(dict1[b'labels'], num_classes)
					if self.y_label is None:
						self.y_label=y_label_temp
					else:
						self.y_label=np.vstack((self.y_label,y_label_temp))
				else:
					y_label_temp=dict1[b'labels']
					if self.y_label is None:
						self.y_label=y_label_temp
					else:
						self.y_label= self.y_label+y_label_temp

		elif split == 'test':			
			testname = os.path.join(data_dir, 'test_batch')
			dicttest = unpickle(testname)
			x_img_temp=dicttest[b'data'].astype(np.float)
			for i in range(number_to_load):
			    self.x_img.append((x_img_temp[i]/255.).reshape((3,32,32)).astype(np.float))
			if as_categorical:
				self.y_label = to_categorical(dicttest[b'labels'], num_classes)
			else:
				self.y_label = dicttest[b'labels']
		self.x_img = np.array(self.x_img)
		self.y_label = np.array(self.y_label)
		self.data_size = self.x_img.shape[0]