from utils.debug_switches import *
from utils.packages import *

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

class DictionaryTxtManager(object):
	def __init__(self, diary_dir, diary_name, dictionary_data=None, header=None):
		super(DictionaryTxtManager, self).__init__()
		self.diary_full_path = os.path.join(diary_dir,diary_name)
		if not os.path.exists(diary_dir): os.mkdir(diary_dir)
		
		self.diary_mode = 'a' 
		if not os.path.exists(self.diary_full_path): self.diary_mode = 'w'

		if dictionary_data is not None:
			self.write_dictionary_to_txt(dictionary_data, header=header)			

	def write_dictionary_to_txt(self, dictionary_data,header=None):
		txt = open(self.diary_full_path,self.diary_mode)
		if header is not None: txt.write(header)
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

class Logger():
	"""
	Usage, run your program after running the following
	sys.stdout = Logger(full_path_log_file="hello.txt")
	"""
	def __init__(self, full_path_log_file="logfile.log"):
		self.terminal = sys.stdout
		self.log = open(full_path_log_file, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self): pass   

def read_csv(filename, header_skip=1,get_the_first_N_rows = 0):
    # filename : string, name of csv file without extension
    # headerskip : nonnegative integer, skip the first few rows
    # get_the_first_N_rows : integer, the first few rows after header_skip to be read
    #   if all rows are to be read, set to zero

    # example
    # xx = read_csv('tryy', 3)
    # xx = [[np.float(x) for x in y] for y in xx] # nice here
    # print(xx)
    # print(np.sum(xx[0]))
    
    out = []
    with open(str(filename)+'.csv', encoding='cp932', errors='ignore') as csv_file:
        data_reader = csv.reader(csv_file)
        count = 0
        i = 0
        for row in data_reader:
            if count < header_skip:
                count = count + 1
            else:
                out.append(row)
                # print(row)
                if get_the_first_N_rows>0:
                	i = i + 1
                	if i == get_the_first_N_rows:
                		break
    return out

def normalize_numpy_array(x,target_min=-1,target_max=1, source_min=None, source_max=None, verbose = 250):
	'''
	If target_min or target_max is set to None, then no normalization is performed
	'''
	if source_min is None: source_min = np.min(x)
	if source_max is None: source_max = np.max(x)
	if target_min is None or target_max is None: return x
	if source_min==source_max:
		if verbose> 249 : print("normalize_numpy_array: constant array, return unmodified input")
		return x
	midx0=0.5*(source_min+source_max)
	midx=0.5*(target_min+target_max)
	y=x-midx0
	y=y*(target_max - target_min )/( source_max - source_min)
	return y+midx
