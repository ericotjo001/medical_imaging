
import pickle
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


fullpath = 'gg.wp'
thisobj = SaveableObject()
thisobj.set_a('777')
thisobj.save_object(fullpath)

thisobj2 = SaveableObject().load_object(fullpath)
print("thisobj2.a:%s"%(thisobj2.a))
		