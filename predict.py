import joblib
import pandas as pd

# Load the trained model and preprocessor
model_file_path = 'rf_model.pkl'
rf_model, preprocessor = joblib.load(model_file_path)

# Load the new data for prediction
new_data_file_path = 'test_data.xlsx'
new_data = pd.read_excel(new_data_file_path)

# Ensure the new data has the same columns as the training data
expected_columns = ['CUSTOMER_ID', 'PRODUCT_CODE', 'STATUSDESCRIPTION', 'COMPONENT_NAME',
                    'Total_Payment_Delays', 'PENALTY_COUNT']

# Check for missing columns
missing_cols = set(expected_columns) - set(new_data.columns)
if missing_cols:
    raise ValueError(f"Missing columns in the new data: {missing_cols}")

# Ensure the order of columns is the same
new_data = new_data[expected_columns]

# Apply the same preprocessing to the new data (excluding CUSTOMER_ID for transformation)
new_data_transformed = preprocessor.transform(new_data.drop(columns=['CUSTOMER_ID']))

# Make predictions
predictions_proba = rf_model.predict_proba(new_data_transformed)
threshold = 0.3  # Adjust the threshold as needed
predictions = (predictions_proba[:, 1] >= threshold).astype(int)

# Classify customers with high Total_Payment_Delays as defaulted
high_delay_threshold = 30  # Example threshold for high payment delays
predictions[new_data['Total_Payment_Delays'] > high_delay_threshold] = 1

# Output the predictions
output = pd.DataFrame({
    'CUSTOMER_ID': new_data['CUSTOMER_ID'],
    'PRODUCT_CODE': new_data['PRODUCT_CODE'],
    'STATUSDESCRIPTION': new_data['STATUSDESCRIPTION'],
    'COMPONENT_NAME': new_data['COMPONENT_NAME'],
    'Total_Payment_Delays': new_data['Total_Payment_Delays'],
    'PENALTY_COUNT': new_data['PENALTY_COUNT'],
    'Prediction': predictions,
    'Probability_Class_0': predictions_proba[:, 0],
    'Probability_Class_1': predictions_proba[:, 1]
})

output_file_path = 'predictions_output.csv'
output.to_csv(output_file_path, index=False)
print("Predictions saved to", output_file_path)
