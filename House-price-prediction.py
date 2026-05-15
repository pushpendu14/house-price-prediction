import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Loaded Dataset
df = pd.read_csv('C:/Users/Pushpendu/Downloads/Housing.csv')

print(df.head())
print(df.shape)
print(df.info())

print(df.isnull().sum())

df = pd.get_dummies(df, drop_first=True)

#Visualization
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# House Price Distribution
sns.histplot(df['price'], kde=True)
plt.title('House Price Distribution')
plt.show()

# Area Vs Price
sns.scatterplot(x=df['area'], y=df['price'])
plt.title('Area vs Price')
plt.show()

# Features and Target

X = df.drop('price', axis=1)
y = df['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Traning
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print('MAE:', mae)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)

# R2 Score
r2 = r2_score(y_test, y_pred)
print('R2 Score:', r2)

# Comparing Actual vs Predicted
comparison = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})
print(comparison.head())

# Visualization of Predictions
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()














