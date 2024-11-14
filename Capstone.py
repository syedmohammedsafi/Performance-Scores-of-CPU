import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 1: Import the dataset
data = pd.read_csv("machine_data.csv")

# Step 2: Data preprocessing
# a. Encode the categorical column 'vendor' using label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['vendor'] = le.fit_transform(data['vendor'])
# b. Identify vendors with less than 5 CPUs and drop rows corresponding to those vendors
vendor_counts = data['vendor'].value_counts()
vendors_to_drop = vendor_counts[vendor_counts < 5].index
data = data[~data['vendor'].isin(vendors_to_drop)]
# c. Drop the column 'model'
data = data.drop('model', axis=1)

# Step 3: Select 'score' as the target variable and remaining features as predictors
X = data.drop('score', axis=1)
y = data['score']

# Step 4: Split the data into training and testing sets (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model building
# a. Build a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# b. Find the train and test scores for the built model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# c. Calculate the adjusted R-squared values
n = X_train.shape[0]
p = X_train.shape[1]
r2_train = 1 - (1 - train_score) * ((n - 1) / (n - p - 1))
r2_test = 1 - (1 - test_score) * ((n - 1) / (n - p - 1))

# Display the results
print("Train Score:", train_score)
print("Test Score:", test_score)
print("Adjusted R-squared (Train):", r2_train)
print("Adjusted R-squared (Test):", r2_test)

# Step 6: Calculate the VIF values
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

vif_values = calculate_vif(X_train)
print("VIF Values:")
print(vif_values)

# Step 7: Plot the regression lines for training set
plt.figure(figsize=(10, 6))
plt.scatter(X_train['cycle_time'], y_train, color='blue', label='Training Data')

# Plot the regression line for training data
plt.plot(X_train['cycle_time'], model.predict(X_train), color='black', linewidth=2, label='Regression Line (Train)')

# Step 8: Plot the regression lines for test set
plt.figure(figsize=(10, 6))
plt.scatter(X_test['cycle_time'], y_test, color='red', label='Test Data')
plt.title('Linear Regression - Test Data')
plt.xlabel('Cycle Time')
plt.ylabel('Score')
plt.legend()
plt.show()

# Step 9: Create a DataFrame to store the results
results = pd.DataFrame({
    'Train Score': [train_score],
    'Test Score': [test_score],
    'Adjusted R2 (Train)': [r2_train],
    'Adjusted R2 (Test)': [r2_test]
})

print(results)

# Step 10: Predict the performance score of new CPU instances
new_cpu_instances = pd.DataFrame({
    'vendor': [14, 0, 0, 0, 0, 0],
    'cycle_time': [125, 29, 29, 29, 29, 26],
    'min_memory': [256, 8000, 8000, 8000, 16000, 32000],
    'max_memory': [6000, 32000, 32000, 32000, 16000, 64000],
    'cache': [256, 32, 32, 32, 32, 64],
    'min_threads': [16, 8, 8, 8, 8, 8],
    'max_threads': [128, 32, 32, 32, 16, 32]
})
predicted_scores = model.predict(new_cpu_instances)
print(predicted_scores)

# Display the predicted scores for new CPU instances
print("\nPredicted Scores for New CPU Instances:")
print(new_cpu_instances)
