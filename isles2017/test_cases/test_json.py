import json, collections, os

class ConfigManager(object):

	def __init__(self):
		super(ConfigManager, self).__init__()
		
	def create_config_file(self,config_dir):
		print("utils/utils.py. create_config_file()")

		temp_dir = "temp.json"
		with open(temp_dir, 'w') as json_file:  
			json.dump(CONFIG_FILE, json_file, separators=(',',":"), indent=2) # (',', ':')

		self.json_beautify(temp_dir, config_dir)
		print("  Config file created.")

	def json_file_to_pyobj(self,filename):
	    def _json_object_hook(d): return collections.namedtuple('ConfigX', d.keys())(*d.values())
	    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
	    return json2obj(open(filename).read())

	def json_file_to_pyobj_recursive_print(self,config_data, name='config_data', verbose=0):
		for field_name in config_data._fields:
			temp = getattr(config_data,field_name)
			if 'ConfigX' in str(type(temp)) : self.json_file_to_pyobj_recursive_print(temp, name+'.'+field_name)
			else: 
				print("  %s.%s: %s [%s]"%(name,field_name, str(temp), str(type(temp))))
				if isinstance(temp,list) and verbose>9: 
					for x in temp: print("      [%s]"%(type(x)),end='')
					print()
				
	def json_beautify(self,temp_dir, config_dir):
		bracket = 0
		
		with open(config_dir, 'w') as json_file: 
			for x in open(temp_dir, 'r'):
				temp = x.strip("\n").strip("\t")
				prev_bracket = bracket
				if "]" in temp: bracket-=1 
				if "[" in temp: bracket+=1
				if bracket ==0 and not prev_bracket==1: print(temp); json_file.write(temp+"\n")
				elif bracket ==0 and prev_bracket==1: print(temp.strip(" ")); json_file.write(temp.strip(" ")+"\n")
				elif bracket==1 and prev_bracket==0: print(temp,end=''); json_file.write(temp)
				else: print(temp.strip(" "),end='') ; json_file.write(temp.strip(" "))
		os.remove(temp_dir)


	def recursive_namedtuple_to_dict(self, config_data):
		if not type(config_data) == type({}):
			config_data = config_data._asdict()
		for xkey in config_data:
			if 'ConfigX' in str(type(config_data[xkey])):
				config_data[xkey] = config_data[xkey]._asdict()
				for ykey in config_data[xkey]:
					if 'ConfigX' in str(type(config_data[xkey][ykey])):
						config_data[xkey][ykey] = self.recursive_namedtuple_to_dict(config_data[xkey][ykey])
		return config_data
CONFIG_FILE = {
	'working_dir':"D:/Desktop@D/meim2venv/meim3",
	'data_dir':{
		'ISLES2017':"D:/Desktop@D/meim2venv/meim3/data/isles2017"
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
			}
		}
	}
}

# Example usage
config_dir = 'config.json'
cm = ConfigManager()
cm.create_config_file(config_dir)

print("\n========= Loading json data =========")
config_data = cm.json_file_to_pyobj(config_dir)
cm.json_file_to_pyobj_recursive_print(config_data, 'config_data')
print("\nThe following demonstrate how to manually access the json data")
print("Example 1. config_data.data_dir.ISLES2017:%s"%(config_data.data_dir.ISLES2017))
x = config_data.some_numbers2
print("Example 2. config_data.some_numbers2:%s"%(x))
print("  x[0]+x[1]=%s+%s = %s"%(x[0],x[1],x[0]+x[1]))

print("\nNote that we can convert config_data (a NamedTuple) to a dictionary, which can be very convenient.")
config_data = cm.recursive_namedtuple_to_dict(config_data)
for xkey in config_data:
	if isinstance(config_data[xkey],dict):
		for ykey in config_data[xkey]:
			print("config_data[%s][%s]: %s"%(str(xkey), str(ykey),str(config_data[xkey][ykey])))
	else:
		print("config_data[%s]: %s"%(str(xkey),str(config_data[xkey])))
