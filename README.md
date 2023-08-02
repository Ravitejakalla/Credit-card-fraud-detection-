# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset (Assuming you have a CSV file with features and labels)
# Replace 'data.csv' with the actual filename and adjust the path accordingly
data = pd.read_csv('data.csv')

# Data preprocessing: Drop irrelevant columns, handle missing values, and normalize features as needed
# Drop 'Time' and 'Amount' columns as they might not be useful for the model
data = data.drop(['Time', 'Amount'], axis=1)

# Split data into features (X) and labels (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

# Confusion matrix to assess model performance
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Check the number of fraudulent transactions in the test set
num_fraudulent_transactions = sum(y_test)
print("Number of Fraudulent Transactions in Test Set: {}".format(num_fraudulent_transactions))

# Check the number of correctly detected fraudulent transactions
true_positive = np.sum((y_test == 1) & (y_pred == 1))
print("True Positives (Correctly Detected Fraudulent Transactions): {}".format(true_positive))

# Check the number of incorrectly detected fraudulent transactions (false positives)
false_positive = np.sum((y_test == 0) & (y_pred == 1))
print("False Positives (Incorrectly Detected Fraudulent Transactions): {}".format(false_positive))

# Check the number of missed fraudulent transactions (false negatives)
false_negative = np.sum((y_test == 1) & (y_pred == 0))
print("False Negatives (Missed Fraudulent Transactions): {}".format(false_negative))

# Calculate precision, recall, and F1-score
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-Score: {:.2f}".format(f1_score))
