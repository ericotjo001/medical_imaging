import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # add parent dir
from utils.utils import *
from sklearn.linear_model import LinearRegression

tis_size = 1000
x = np.linspace(0,10,tis_size) + np.random.normal(0,2,size=(tis_size,))
y = 10 - 2 * x + np.random.normal(0,5,size=(tis_size,))

X = [[x1] for x1 in x]
reg = LinearRegression().fit(X, y)
lin_reg = reg.score(X, y)
print('lin_reg:%s'%(str(lin_reg)))
print('reg.coef_:%s'%(str(reg.coef_)))
print('reg.intercept_:%s'%(str(reg.intercept_)))

m, c = reg.coef_[0], reg.intercept_
x0 = np.linspace(np.min(x), np.max(x), 250)
y_pred = m*x0+c

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x,y,s=3,marker='x')
ax.plot(x0,y_pred)
plt.show()


