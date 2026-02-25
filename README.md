# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data = pd.read_csv('CarPrice_Assignment.csv')
data = data.drop(['car_ID', 'CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
print('Name: ')
print('Name: ')
print('Reg. No: ')
print("\n=== Cross-Validation ===")
Name: 
Reg. No: 

=== Cross-Validation ===
cv_scores = cross_val_score(model, X, y, cv=5)
print("Fold R^2 scores:", [f"{score:.4f}" for score in cv_scores])
print(f"Average R^2: {cv_scores.mean():.4f}")
Fold R^2 scores: ['0.6238', '0.6316', '0.3132', '0.3643', '-0.4944']
Average R^2: 0.2877
y_pred = model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Average R²: {cv_scores.mean():.4f}")
​
y_pred = model.predict(X_test)
​
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
​
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

=== Test Set Performance ===
MSE: 8482008.48
Average R²: 0.2877

=== Test Set Performance ===
MSE: 8482008.48
R²: 0.8926

/*
Program to implement SGD Regressor for linear regression.
Developed by: Roopika.m
RegisterNumber:212225040348  
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="713" height="545" alt="image" src="https://github.com/user-attachments/assets/6a6fa92c-7b5a-437e-8366-cab1efe0cbe1" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
