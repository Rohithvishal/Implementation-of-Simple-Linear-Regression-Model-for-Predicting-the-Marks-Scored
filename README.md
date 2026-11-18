# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

# AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

1. Start the program.
2. Load the student scores dataset.
3. Separate `Hours` and `Scores`.
4. Split the data into training and testing sets.
5. Create and train the Linear Regression model.
6. Predict student marks.
7. Evaluate the model using MAE, MSE, and RMSE.
8. Plot the regression line and predicted scores.
9. Display the results.
10. Stop the program.

# Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
```
## Step 1: Load the dataset
```
df = pd.read_csv("exp_2_dataset_student_scores.csv")

print("DATASET")
print(df)
```
## Step 2: Display the first and last five rows
```
print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())
```
## Step 3: Separate input and output variables
```
X = df[["Hours"]]
Y = df["Scores"]

print("\nInput Variable (Hours):")
print(X)
print("\nOutput Variable (Scores):")
print(Y)
```
## Step 4: Split the dataset into training and testing sets
```
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0
)

print("\nNumber of Training Samples:", len(X_train))
print("Number of Testing Samples:", len(X_test))
```
## Step 5: Create and train the linear regression model
```
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print("\nLinear Regression Model")
print("Slope:", regressor.coef_[0])
print("Intercept:", regressor.intercept_)
```

## Step 6: Predict scores for the testing data
```
Y_pred = regressor.predict(X_test)

print("\nActual Scores:")
print(Y_test.values)

print("\nPredicted Scores:")
print(np.round(Y_pred, 2))
```
## Step 7: Plot the training data and regression line
```
plt.figure(figsize=(8, 5))
plt.scatter(X_train["Hours"], Y_train, color="blue",
            label="Training Data")

line_x = np.linspace(df["Hours"].min(), df["Hours"].max(), 100)
line_y = regressor.predict(pd.DataFrame({"Hours": line_x}))

plt.plot(line_x, line_y, color="red",
         label="Regression Line")

plt.title("Hours Studied vs Student Scores - Training Data")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```
## Step 8: Plot actual and predicted testing scores
```
plt.figure(figsize=(8, 5))
plt.scatter(X_test["Hours"], Y_test, color="blue",
            label="Actual Scores")
plt.scatter(X_test["Hours"], Y_pred, color="red",
            marker="x", s=80, label="Predicted Scores")

plt.title("Actual vs Predicted Student Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```
## Step 9: Evaluate model performance
```
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("\nMODEL EVALUATION")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
```
## Step 10: Predict the score for a new study duration
```
new_hours = pd.DataFrame({"Hours": [9.25]})
predicted_score = regressor.predict(new_hours)

print("\nPREDICTION FOR A NEW STUDENT")
print("Hours Studied:", 9.25)
print("Predicted Score:", round(predicted_score[0], 2))

print("\nRESULT: Simple Linear Regression was implemented successfully.")

```

# Output:
<img width="982" height="602" alt="Screenshot 2026-10-26 132326" src="https://github.com/user-attachments/assets/f933d814-eb5e-46e5-a8e9-a0110eb762de" />
<img width="966" height="626" alt="Screenshot 2026-10-26 132635" src="https://github.com/user-attachments/assets/d7799774-1744-4245-b015-ff5aa509e276" />

# Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
