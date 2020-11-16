from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)

X = [[1, 2, 3], [11, 12, 13]]
y = [0, 1]
clf.fit(X, y)

clf.predict([[4,5,6], [14,15,16]])


from sklearn.preprocessing import StandardScaler
X = [[0,15],[1, -10]]
StandardScaler().fit(X).transform(X)


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=0)
)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe.fit(X_train, y_train)


accuracy_score(pipe.predict(X_test), y_test)


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0) # this is cool! it generates a random regression problem
lr = LinearRegression()

result = cross_validate(lr, X, y)
result['test_score']


from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

param_distributions = {'n_estimators': randint(1, 5),
                        'max_depth': randint(5, 10)}
                
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                            n_iter=5,
                            param_distributions=param_distributions,
                            random_state=0)

search.fit(X_train, y_train)

search.best_params_


search.score(X_test, y_test)



