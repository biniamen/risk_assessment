import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Define the numerical and categorical columns again
categorical_cols = ['PRODUCT_CODE', 'P_ADDRESS1']
numerical_cols = ['Total_Amount_Financed', 'Total_Amount_Repaid', 'Number_of_Loans',
                  'Average_Loan_Tenure', 'Total_Payment_Delays',
                  'Total_Interest_Paid', 'Average_Principal_Interest_Ratio']

# Load the preprocessed data with the fitted transformer
file_path = 'preprocessed_data_with_transformer.pkl'
X_train, X_test, y_train, y_test, preprocessor = joblib.load(file_path)

# Initialize the model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Save the trained model and preprocessor
model_file_path = 'rf_model.pkl'
joblib.dump((rf_model, preprocessor), model_file_path)
print("Model training complete and saved to", model_file_path)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Random Forest - Accuracy: {accuracy}")
print(f"Random Forest - Precision: {precision}")
print(f"Random Forest - Recall: {recall}")
print(f"Random Forest - F1 Score: {f1}")

# Feature importance
feature_importances = rf_model.feature_importances_

# Get the feature names after preprocessing
num_features = preprocessor.named_transformers_['num'].get_feature_names_out(numerical_cols)
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
feature_names = np.array(list(num_features) + list(cat_features))

# Exclude 'P_ADDRESS1' from the feature names and importance
exclude_feature = 'P_ADDRESS1'
include_indices = feature_names != exclude_feature
filtered_feature_names = feature_names[include_indices]
filtered_feature_importances = feature_importances[include_indices]

# Plot feature importance
plt.figure(figsize=(12, 10))  # Increase the figure size
sorted_idx = np.argsort(filtered_feature_importances)
plt.barh(filtered_feature_names[sorted_idx], filtered_feature_importances[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.title("Feature Importance")
plt.tight_layout()  # Adjust the layout to ensure labels are fully visible
plt.show()
