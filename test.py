from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
X=[]
for i in range(0, 25):
        asd=(1,i)
        X.append(asd)

y = [4.83428621, 4.46500015, 5.125, 6.2125001, 6.74107122, 5.98000002, 5.67678595, 6.05464315, 4.05928612, 3.84249997,
     3.30964303, 3.04821396, 3.21892905, 3.18964291, 3.75428605, 4.49392891, 4.85035706, 5.08678579, 5.83535719,
     6.00750017, 6.61964321, 6.73214293, 7.13964319, 7.52607107, 6.85928583]

kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,normalize_y=True,random_state=0).fit(X, y)
gpr.score(X, y)
print(gpr.predict([(1,i)]))

x2=[]
y2= [5.08678579, 5.83535719,6.00750017, 6.61964321, 6.73214293, 7.13964319, 7.52607107, 6.85928583]
for i in range(0, 8):
        asd=(1,i)
        x2.append(asd)

gpr = GaussianProcessRegressor(kernel=kernel,normalize_y=True,random_state=0).fit(x2, y2)
print(gpr.score(x2, y2))
print(gpr.predict([(1,8)]))