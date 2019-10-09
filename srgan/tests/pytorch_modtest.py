import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils.utils import *
from torch.autograd import Function

def main():

	class Quadratic(nn.Module):
		"""docstring for Quadratic"""
		def __init__(self):
			super(Quadratic, self).__init__()

			self.weight = torch.nn.Parameter(data=torch.Tensor([1, 2.4, 3]), requires_grad=True)
		def forward(self,x):
			v = torch.tensor([x**2,x,1])
			return torch.sum(self.weight*v)

	a,b,c=1,2,3
	X = np.random.normal(0,1,size=(10))
	X = torch.tensor(X, requires_grad=True).to(torch.float)
	Y0 = X**2
	Y_set = []

	net = Quadratic()
	criterion = nn.MSELoss()
	optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

	print(net.weight.data)

	for i in range(4):
		Y = []
		for j in range(len(X)):
			optimizer.zero_grad()
			y = net(X[j])
			# print(y.item())
			Y.append(y.item())
			loss = criterion(y,Y0[j])
			loss.backward(retain_graph=True)
			optimizer.step()
		Y_set.append(Y)

	print(net.weight.data)

	fig = plt.figure()
	
	for Y in Y_set:
		plt.scatter(X.detach().numpy(),Y)
		print(Y)
	ax2 = plt.scatter(X.detach().numpy(),Y0.detach().numpy(),marker='x')
	plt.show()

if __name__=="__main__":
	main()