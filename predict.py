import joblib
import pandas as pd

# Load the trained model and preprocessor
model_file_path = 'rf_model.pkl'
rf_model, preprocessor = joblib.load(model_file_path)

# Load the new data for prediction
new_data_file_path = 'test_data.xlsx'
new_data = pd.read_excel(new_data_file_path)

# Ensure the new data has the same columns as the training data
expected_columns = ['PRODUCT_CODE', 'P_ADDRESS1', 'Total_Amount_Financed', 
                    'Total_Amount_Repaid', 'Number_of_Loans', 'Average_Loan_Tenure', 
                    'Total_Payment_Delays', 'Total_Interest_Paid', 'Average_Principal_Interest_Ratio']

# Check for missing columns
missing_cols = set(expected_columns) - set(new_data.columns)
if missing_cols:
    raise ValueError(f"Missing columns in the new data: {missing_cols}")

# Ensure the order of columns is the same
new_data = new_data[expected_columns]

# Apply the same preprocessing to the new data
new_data_transformed = preprocessor.transform(new_data)

# Make predictions
predictions = rf_model.predict(new_data_transformed)
predictions_proba = rf_model.predict_proba(new_data_transformed)

# Output the predictions
output = pd.DataFrame({
    'Prediction': predictions,
    'Probability_Class_0': predictions_proba[:, 0],
    'Probability_Class_1': predictions_proba[:, 1]
})

output_file_path = 'predictions.csv'
output.to_csv(output_file_path, index=False)
print("Predictions saved to", output_file_path)
