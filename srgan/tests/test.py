from utils.utils import * 

from dataio.cifar2loader import Cifar10Data

def test(config_dir):
	config_data = get_config_data(config_dir)

	if config_data['debug_test_mode'] == 'test_load_cifar': test_load_cifar(config_data)
	
def test_load_cifar(config_data):
	print("test/test.py. test_load_cifar()")
	data_dir = config_data['data_directory']['cifar10']
	split = 'train'
	cf = Cifar10Data()
	cf.size_per_raw_batch = 100 # default = 10000
	cf.load_data(config_data)
	# cf2 = Cifar10Data()
	# cf2.load_data(config_data)

	print("data_dir:",data_dir)
	print("cf.x_img.shape  : ", cf.x_img.shape)
	print("cf.y_label.shape: ", cf.y_label.shape)
	print("  cf.__len__() : ",cf.__len__())
	n = 0
	x_img1 , y_label1 = cf.__getitem__(n+0)
	x_img2 , y_label2 = cf.__getitem__(n+1)
	x_img3 , y_label3 = cf.__getitem__(n+2)
	x_img4 , y_label4 = cf.__getitem__(n+3)
	print("  x_img1.shape = %s, y_label1 = %s: "%(str(x_img1.shape), str(y_label1)))
	print("  x_img2.shape = %s, y_label2 = %s: "%(str(x_img2.shape),str(y_label2)))
	# print("cf2.x_img.shape  : ", cf2.x_img.shape)
	# print("cf2.y_label.shape: ", cf2.y_label.shape)
	# print("  cf2.__len__() : ",cf2.__len__())

	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	ax1.set_title(str(y_label1)+":"+str(cf.img_dict[y_label1])) # self.img_dict[i]
	plt.imshow(x_img1.transpose(1,2,0))
	
	ax2 = fig.add_subplot(222)
	ax2.set_title(str(y_label2)+":"+str(cf.img_dict[y_label2]))
	plt.imshow(x_img2.transpose(1,2,0))
	
	ax3 = fig.add_subplot(223)
	ax3.set_title(str(y_label3)+":"+str(cf.img_dict[y_label3]))
	plt.imshow(x_img3.transpose(1,2,0))

	ax4 = fig.add_subplot(224)
	ax4.set_title(str(y_label4)+":"+str(cf.img_dict[y_label4]))
	plt.imshow(x_img4.transpose(1,2,0))
	
	plt.tight_layout()
	plt.show()

	cf2 = Cifar10Data()
	cf2.size_per_raw_batch = 100 # default = 10000
	cf2.load_data(config_data, split = 'test')
	x_img_test_1 , y_label_test_1 = cf2.__getitem__(0)
	x_img_test_2 , y_label_test_2 = cf2.__getitem__(1)
	x_img_test_3 , y_label_test_3 = cf2.__getitem__(2)
	x_img_test_4 , y_label_test_4 = cf2.__getitem__(3)

	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	ax1.set_title(str(y_label_test_1)+":"+str(cf2.img_dict[y_label_test_1])) # self.img_dict[i]
	plt.imshow(x_img_test_1.transpose(1,2,0))
	
	ax2 = fig.add_subplot(222)
	ax2.set_title(str(y_label_test_2)+":"+str(cf2.img_dict[y_label_test_2]))
	plt.imshow(x_img_test_2.transpose(1,2,0))

	ax3 = fig.add_subplot(223)
	ax3.set_title(str(y_label_test_3)+":"+str(cf2.img_dict[y_label_test_3]))
	plt.imshow(x_img_test_3.transpose(1,2,0))

	ax4 = fig.add_subplot(224)
	ax4.set_title(str(y_label_test_4)+":"+str(cf2.img_dict[y_label_test_4]))
	plt.imshow(x_img_test_4.transpose(1,2,0))
	
	plt.tight_layout()
	plt.show()