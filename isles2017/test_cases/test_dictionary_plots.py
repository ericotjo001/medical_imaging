import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *
from utils.vis import *

'''
From the main working directory
python tests/test_dictionary_plots
'''

dm = DictionaryWithNumericalYArray()
DEBUG_MODE = 2
if DEBUG_MODE ==0:
	dm.this_dictionary = dm.get_example(number_of_classes=2, verbose=10)
	dm.auto_get_mapping_index()
	dm.get_plainlist(verbose=10)
	# dm.scatter_plainlist(xlim=[-1,2],ylim=[-3,4])
	dm.box_plainlist()
	plt.show()

elif DEBUG_MODE == 1:
	dm.this_dictionary = dm.get_example(number_of_classes=5, size=50, verbose=0)
	dm.normal_scatter_mapping_index()
	dm.get_normal_scatter_list(mu=0,sigma=0.2, verbose=10)
	dm.scatter_normal_scatter_list(xlim=[-1,5],ylim=[-3,7])
	plt.show()

elif DEBUG_MODE == 2:
	# collection = {}
	# for i in range(1):
	# 	collection_name = 'p'+str(i)
	# 	data = []
	# 	print(collection_name)
	# 	for j in range(1):
	# 		dm.this_dictionary = dm.get_example(number_of_classes=2, verbose=0)
	# 		dm.auto_get_mapping_index()
			# dm.get_plainlist(verbose=0)
			# print(dm.this_dictionary)
			# data.append(dm.plainlist)
		# collection[collection_name] = data
	# dm.layered_box_plainlist(collection)
	model_label_names=[
		'UNet3D_AXXXS1',
		'UNet3D_AXXXS2'
	]

	model_label_names_collection = {
		'Collection1': model_label_names,
		'Collection2': ['Anet'],
		'Collection3': ['AXnet']
	}



	dict_of_XY = {}
	for collection_name in model_label_names_collection:
		model_label_names = model_label_names_collection[collection_name]
		# print('collection_name:%s'%(str(collection_name)))
		XY = {}
		for model_label_name in model_label_names:
			for i in range(4):
				XY['x'+str(i)] = np.random.randint(0,10,10)
		
		# for xkey in XY:
		# 	print('  xkey=%s\n    %s'%(str(xkey), str(XY[xkey])))
		
		dict_of_XY[collection_name] = XY
	
	color_list = {
		'Collection1': 'g',
		'Collection2': 'r',
		'Collection3': 'b'

	}
	dm.layered_boxplots(dict_of_XY, color_list, shift_increment=0.75, x_index_increment=4)
	plt.show()