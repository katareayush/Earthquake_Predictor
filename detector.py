import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('dataset.csv')

X = data[['Latitude', 'Longitude', 'Depth']]
y = data['Magnitude']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

num_samples = 10
random_samples = data.sample(n=num_samples, random_state=42)

X_random = random_samples[['Latitude', 'Longitude', 'Depth']]
y_random_pred = best_model.predict(X_random)

random_samples['Predicted Magnitude'] = y_random_pred

mean_predicted = np.mean(y_random_pred)
variance_predicted = np.var(y_random_pred)
std_dev_predicted = np.std(y_random_pred)

print(f'Random Sample Predictions:\n{random_samples[["Latitude", "Longitude", "Depth", "Predicted Magnitude"]]}\n')
print(f'Mean Predicted Magnitude: {mean_predicted}')
print(f'Variance of Predicted Magnitude: {variance_predicted}')
print(f'Standard Deviation of Predicted Magnitude: {std_dev_predicted}')

print("\nFirst 10 Actual vs Predicted Values:")
print(results.head(10))
