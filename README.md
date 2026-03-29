# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Start

Import required libraries

Load the dataset

Preprocess the data (handle missing values, separate X and y, perform feature scaling)

Split the dataset into training and testing sets

Initialize the SGD Regressor model

Train the model using the training data

Predict the output using the test data

Evaluate the model performance using MSE, RMSE, and R² score

Display the results

Stop
``` 

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, y_train)
C:\ProgramData\anaconda3\lib\site-packages\sklearn\utils\validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
SGDRegressor()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_pred = sgd_model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("Name:Roopika m")
print("Reg no:25008774")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:",mae)
print("R-squared Score:", r2)
Name:Roopika m
Reg no:25008774
Mean Squared Error: 0.21025456651198762
Mean Absolute Error: 0.3125189308265761
R-squared Score: 0.8308502781730002
print("\nModel Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()


/*
Program to implement SGD Regressor for linear regression.
Developed by: Roopika.m
RegisterNumber:212225040348  
*/
```

## Output:


![simple linear regression model for predicting the marks scored](sam.png)
<img width="897" height="532" alt="image" src="https://github.com/user-attachments/assets/a53d6b87-dacd-491a-bd69-f4ed30158870" />
<img width="541" height="682" alt="image" src="https://github.com/user-attachments/assets/bebb35a8-310c-4c91-9a05-b075b61b2551" />

<img width="797" height="210" alt="image" src="https://github.com/user-attachments/assets/2e2e1521-8439-4417-8779-d9953ba52b5d" />
<img width="565" height="453" alt="image" src="https://github.com/user-attachments/assets/9f13fa1e-e4f8-4b9b-985e-32b171b3e05b" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
