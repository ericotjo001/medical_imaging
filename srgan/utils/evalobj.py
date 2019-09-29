from utils.utils import *

class LossTracker(object):
	"""docstring for LossTracker"""
	def __init__(self, config_data, display_every_n_minibatchs=10):
		super(LossTracker, self).__init__()
		self.display_every_n_minibatchs = display_every_n_minibatchs
		self.loss_name = 'someLoss'

		model_dir = os.path.join(config_data['working_dir'], config_data['relative_checkpoint_dir'],config_data['model_label_name'])
		self.model_dir = model_dir
		self.plot_fullpath = os.path.join(model_dir,config_data['model_label_name'] +"_loss_temp" + '.jpg')
		self.plot_fullpath_without_extension = os.path.join(model_dir,config_data['model_label_name'] +"_loss_")
		
		self.running_count= 0
		self.running_error = []
		self.running_loss = 0.
		self.global_count = 0
		
		self.avg_loss = []
		self.loss_var = []
		self.k_th_global_step = []

	def store_loss(self, loss):
		self.running_loss = self.running_loss + float(loss.item())
		self.running_error.append(float(loss.item()))
		self.running_count = self.running_count + 1
		self.global_count = self.global_count + 1

		if self.running_count ==  self.display_every_n_minibatchs:
			self.avg_loss.append(self.running_loss/self.display_every_n_minibatchs)
			self.loss_var.append(np.var(self.running_error)**0.5)
			self.k_th_global_step.append(self.global_count)
			self.running_count= 0
			self.running_loss = 0.
			self.running_error = []
			self.save_loss_plot()

	def save_loss_plot(self, label_tag=None):
		error_signal = 0
		first_error_flag = 1
		# print("save_loss_plot(). Plot size:%s"%(str(len(self.k_th_global_step))))
		plt.figure()
		plt.plot(self.k_th_global_step,self.avg_loss)
		y = np.array(self.avg_loss)
		er = np.array(self.loss_var) + 1e-6
		plt.fill_between(self.k_th_global_step,y-er, y+er,color='r',alpha=0.2)
		plt.title(self.loss_name)
		plt.tight_layout()
		while True:
			if error_signal:
				if first_error_flag:
					print("utils/evalobj.py. LossTracker. In error loop.")
					first_error_flag = 0
			try:
				if label_tag is not None: plt.savefig(self.plot_fullpath_without_extension + str(label_tag) + '.jpg')
				else: plt.savefig(self.plot_fullpath)
				break
			except:
				error_signal = 1
		plt.close()


class CrossEntropyLossTracker(LossTracker):
	"""docstring for CrossEntropyLossTracker"""
	def __init__(self, config_data, display_every_n_minibatchs=10):
		super(CrossEntropyLossTracker, self).__init__(config_data, display_every_n_minibatchs=display_every_n_minibatchs)
		self.loss_name = 'CrossEntropyLoss'
		self.plot_fullpath = os.path.join(self.model_dir,config_data['model_label_name'] +"_" + self.loss_name+"_temp" + '.jpg')
		self.plot_fullpath_without_extension = os.path.join(self.model_dir,config_data['model_label_name'] +"_" + self.loss_name+ "_")

class MSELosstracker(LossTracker):
	"""docstring for MSELosstracker"""
	def __init__(self, config_data, display_every_n_minibatchs=10):
		super(MSELosstracker, self).__init__(config_data, display_every_n_minibatchs=display_every_n_minibatchs)
		self.loss_name = 'MSELoss'
		self.plot_fullpath = os.path.join(self.model_dir,config_data['model_label_name'] +"_" + self.loss_name+"_temp" + '.jpg')
		self.plot_fullpath_without_extension = os.path.join(self.model_dir,config_data['model_label_name'] +"_" + self.loss_name+ "_")
		

class PredictionAcc(object):
	def __init__(self):
		super(PredictionAcc, self).__init__()
		self.no_of_correct = 0
		self.total_number_of_data_evaluated = 0

	def setup_classification_matrix(self,C):
		# C is int. No of classes.
		self.cm = np.zeros(shape=(C,C), dtype=int)

	def update_prediction_acc_0001(self, y, y_ot):
		"""
		y is output predicted by network. int, raw input, same format as pytprch CrossEntropyLoss
			e.g. y==0, y==8
		y_ot is the groundtruth. int. Same format as y.
		"""
		if y==y_ot: 
			self.no_of_correct += 1
		self.cm[y][y_ot] += 1


	def process_evaluation_0001(self, model_dir, model, report_name='eval_report'):
		print("PredictionAcc() .process_evaluation_0001()")
		eval_result_fullpath = os.path.join(model_dir, report_name + "_" + str(model.training_cycle) + '.txt') 
		print("  eval_result_fullpath:%s"%(eval_result_fullpath))
		
		txt = open(eval_result_fullpath,'w')
		acc = self.no_of_correct / self.total_number_of_data_evaluated
		print("  acc = %s = %s percent [total n evaluated = %s]"%(str(acc),str(acc*100),str(self.total_number_of_data_evaluated)))
		txt.write("  acc = %s = %s percent [total n evaluated = %s]\n"%(str(acc),str(acc*100),str(self.total_number_of_data_evaluated)))
		self.print_classification_matrix(txt)

		# print("  model_dir: %s"%(model_dir))
		txt.close()

	def print_classification_matrix(self, txt):
		temp = "  classification matrix. cm[i][j] means prediction is row i, groundtruth is column 1"
		print(temp)
		txt.write(temp+"\n")
		print("    %2s | "%(str('')), end=' ')
		txt.write("    %2s | "%(str('')))
		for j, y in enumerate(self.cm[0]):
			print("%5s"%(str(j)), end=' ')
			txt.write("%5s"%(str(j)))
		print("\n    "+"-"*80)
		txt.write("\n    "+"-"*60+"\n")
		for i, row in enumerate(self.cm):
			print("    %2s | "%(str(i)), end=' ')
			txt.write("    %2s | "%(str(i)))
			for j, y in enumerate(self.cm[i]):
				print("%5s"%(str(y)), end=' ')
				txt.write("%5s"%(str(y)))
			print("")
			txt.write("\n")

		