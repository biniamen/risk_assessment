import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Define the numerical and categorical columns again
categorical_cols = ['STATUSDESCRIPTION', 'COMPONENT_NAME']
numerical_cols = ['Total_Payment_Delays', 'PENALTY_COUNT']

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
def get_feature_names(preprocessor):
    num_features = preprocessor.transformers_[0][1].named_steps['scaler'].get_feature_names_out(numerical_cols)
    cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out()
    return np.concatenate([num_features, cat_features])

feature_names = get_feature_names(preprocessor)

# Map encoded feature names back to their original names
original_names_mapping = {
    'x0': 'PRODUCT_CODE',
    'x1': 'STATUSDESCRIPTION',
    'x2': 'COMPONENT_NAME'
}

descriptive_feature_names = []
for feature in feature_names:
    prefix = feature.split('_')[0]
    if prefix in original_names_mapping:
        descriptive_feature_names.append(f"{original_names_mapping[prefix]}_{feature.split('_')[1]}")
    else:
        descriptive_feature_names.append(feature)

# Print feature names to verify
print("Feature names:", descriptive_feature_names)

# Filter out only the relevant features
relevant_features = descriptive_feature_names

# Get relevant feature importances
relevant_importances = [feature_importances[feature_names.tolist().index(f)] for f in feature_names]

# Select top 10 most important features
top_n = 10
sorted_idx = np.argsort(relevant_importances)[-top_n:]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(np.array(relevant_features)[sorted_idx], np.array(relevant_importances)[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.title("Top Feature Importance")
plt.tight_layout()  # Adjust the layout to ensure labels are fully visible
plt.show()
