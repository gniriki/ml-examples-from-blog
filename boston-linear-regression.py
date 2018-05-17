import sklearn.datasets as ds
import sklearn.model_selection as md
from sklearn.preprocessing import PolynomialFeatures

data = ds.load_boston();

data.data = PolynomialFeatures(degree=3).fit_transform(data.data)

X_train, X_test, y_train, y_test = md.train_test_split(
     data.data, data.target, random_state=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import RidgeCV

alphas=[0.1,0.3, 1,3, 10,30,100,300, 1000]
reg = RidgeCV(alphas)

reg.fit(X_train, y_train)

print("Train score: " + str(reg.score(X_train, y_train)))
print("Test score: " + str(reg.score(X_test, y_test)))
print("Selected alpha: " + str(reg.alpha_))

from sklearn.model_selection import learning_curve
from matplotlib import pyplot
import numpy as np

train_sizes, train_scores, cv_scores = learning_curve(reg, X_train, y_train, train_sizes = np.linspace(.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
cv_scores_mean = np.mean(cv_scores, axis=1)

pyplot.ylim((-1, 1.0))

pyplot.plot(train_sizes, train_scores_mean, 'ro-', label="Train score")
pyplot.plot(train_sizes, cv_scores_mean, 'go-', label="CV score")
pyplot.legend()