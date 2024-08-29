from sklearn.linear_models import LogisticRegression
from sklearn.datasets import iris
iris = load_iris()
X = iris.data
y = iris.target
reg = LogisticRegression()
reg.fit(X, y)
reg.predict([[2., 2., 2., 2.]])