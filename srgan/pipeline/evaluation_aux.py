from utils.utils import *


def evaluate_small_cnn_debug0001(outputs, y_label, labels):
	print('evaluate_small_cnn_debug0001()')
	print("  outputs:%s\n  y_label:%s <%s>"%(str(outputs),str(y_label), str(type(y_label))))
	print("  labels (groundtruth):%s <%s>"%(str(labels),str(type(labels))))
	print("  labels ==  y_label ?? %s"%(str(labels == y_label)))

