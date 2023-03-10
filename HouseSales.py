# dataset -> https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
 # Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline
sns.set()

df=pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

df.head()

print(df.dtypes)

counts = df['floors'].value_counts().to_frame()
print(counts)


sns.boxplot(x='waterfront', y='price', data=df)

plt.title('Price distribution by Waterfront View')
plt.xlabel('Waterfront View')
plt.ylabel('Price')

plt.show()


sns.regplot(x='sqft_above', y='price', data=df)

# Add a title and axis labels to the plot
plt.title('Price vs. sqft_above')
plt.xlabel('sqft_above')
plt.ylabel('Price')

# Show the plot
plt.show()


df.corr()['price'].sort_values()


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df[['sqft_living']]
y = df['price']
model = LinearRegression()

model.fit(X, y)
r2 = r2_score(y, model.predict(X))
print('R^2:', r2)

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X = df[features]     
y = df['price']
model=LinearRegression()
model.fit(X,y)
r2 = r2_score(y, model.predict(X))
print('R^2: ', r2)


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]  
model = Pipeline([('regressor', LinearRegression())])
X = df[features]
y = df['price']
model.fit(X, y)
r2 = r2_score(y, model.predict(X))
print('R^2:', r2)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1)
ridge.fit(x_train, y_train)
y_pred = ridge.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('R^2:', r2)


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
ridge = Ridge(alpha=0.1)
ridge.fit(x_train_poly, y_train)
y_pred = ridge.predict(x_test_poly)
r2 = r2_score(y_test, y_pred)
print('R^2:', r2)
