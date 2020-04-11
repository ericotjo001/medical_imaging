import os, sys, collections

CONFIG_FILE = {
	'working_dir':"D:/Desktop@D",
	'data_dir':{
		'ISLES2017':"D:/Desktop@D"
	},
	'some_numbers': [[ 1, 2, 3, 4],[ 1, 2]],
	'some_numbers2': [ 1.2, 3, 'ggwp'],
	'ggw3': [[2,3],[4,4,4],['a','b','c']],
	'some_dict': {
		'a':1,
		'b':177,
		'c':{
			'c0000001':1999,
			'c0000002':{
				'cMMMM': 7777,
				'cAAAA': 'ggwp'
			},
		'some OrderedDict':collections.OrderedDict({'odod':'odod'})
		}
	}
}

dictionary_data = CONFIG_FILE

class DictionaryTxtManager(object):
	def __init__(self, diary_dir, diary_name, dictionary_data=None):
		super(DictionaryTxtManager, self).__init__()
		self.diary_full_path = os.path.join(diary_dir,diary_name)
		if not os.path.exists(diary_dir): os.mkdir(diary_dir)
		
		self.diary_mode = 'a' 
		if not os.path.exists(self.diary_full_path): self.diary_mode = 'w'

		if dictionary_data is not None:
			self.write_dictionary_to_txt(dictionary_data)			

	def write_dictionary_to_txt(self, dictionary_data):
		txt = open(self.diary_full_path,self.diary_mode)
		self.recursive_txt(txt,dictionary_data,space='')
		txt.close()

	def recursive_txt(self,txt,dictionary_data,space=''):
		for x in dictionary_data:
			if isinstance(dictionary_data[x],dict) or type(dictionary_data[x])==type(collections.OrderedDict({})):
				txt.write("%s%s :\n"%(space,x))
				# self.write_dictionary_to_txt(dictionary_data[x],space=space+'  ')
				# for y in dictionary_data[x]:
				# 	txt.write("%s%s :"%(space,y))
				self.recursive_txt(txt,dictionary_data[x], space=space+'  ')
					# txt.write("\n")
			else:
				txt.write("%s%s : %s [%s]"%(space,x, dictionary_data[x],type(dictionary_data[x])))

			txt.write("\n")

diary_dir = os.getcwd()
diary_name = 'diary.txt'

dtm = DictionaryTxtManager(diary_dir,diary_name,dictionary_data=dictionary_data)
print("dtm.diary_full_path:",dtm.diary_full_path)


