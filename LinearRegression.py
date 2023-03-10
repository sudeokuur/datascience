# dataset -> https://www.kaggle.com/datasets/tanuprabhu/linear-regression-dataset

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv("https://www.kaggle.com/datasets/tanuprabhu/linear-regression-dataset/Linear Regression - Sheet1.csv")

df.describe()

# We use corr() function to compute pairwise correlation of columns in the dataframe. The correlation coefficient is a statistical measure that indicates the degree of linear relationship between two variables.
df.corr()


# I want to see that if the dataframe has any null variables that could make my job harder. Use isnull() function to check this situation. (We don't have any null data.)
df.isnull()

#I created two variables as X and Y to making my job easy. I assigned dataframe's X column for X variable, and Y column for Y variable.
X = df["X"]
Y = df["Y"]

#Visualize the data to see the relationship between the independent Y variable and dependent X variable.

plt.scatter(X, Y)

# This was an easy & quick method to draw a graph. Now, we'll write a code with matplotlib library to see a different & more explanated graph for those variables.
plt.figure(figsize = (10,10))
plt.scatter(X, Y)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.title('Graph', fontsize=18)
plt.show()

#Training & Testing the Data

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

model = LinearRegression()

X = df["X"].values
X_all = X.reshape(-1,1)

model.fit(X_all, Y)

y_pred = model.predict(X_all)
y_pred[200], Y[200]


print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')
print(f'MSE: {np.sqrt(mean_squared_error(Y, y_pred))}')
print(f'R-squared score: {r2_score(Y, y_pred)}')

X_train, X_test, y_train, y_test = train_test_split(X_all, Y, train_size=0.7, test_size=0.3, 
                                                    random_state=21, shuffle=True)

plt.scatter(X_train, y_train)
plt.title('Train data', fontsize=18)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()


plt.scatter(X_test, y_test)
plt.title('Test data', fontsize=18)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()


model_train_test = LinearRegression()
model_train_test.fit(X_train, y_train)
y_pred_traintest = model_train_test.predict(X_test)
y_test[20], y_pred_traintest[20]


print(f'R-squared score: {r2_score(y_test, y_pred_traintest)}')
print(f'MSE: {np.sqrt(mean_squared_error(y_test, y_pred_traintest))}')
print(f'Coefficient: {model_train_test.coef_}')
print(f'Intercept: {model_train_test.intercept_}')

df.tail()
# We can see that the last two rows of the dataset could be wrong. We can drop those two variables to predict better

df.corr()
df.drop([299, 298], axis=0)


plt.figure(figsize=(10,10))
plt.scatter(X_all, Y)
plt.plot(X_test, y_test, color='pink')
plt.plot(X_test, y_pred_traintest, color='blue')
plt.title('Compare the models', fontsize=18)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()
