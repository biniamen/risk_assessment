import joblib
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('dataset/my_dataset.csv')

# Convert date columns to datetime
date_columns = ['BOOK_DATE', 'MATURITY_DATE', 'DUE_DATE', 'PAID_DATE', 'DATE_OF_BIRTH']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], format='%m/%d/%Y')

# Ensure CUSTOMER_ID is a 7-digit value
df['CUSTOMER_ID'] = df['CUSTOMER_ID'].apply(lambda x: str(x).zfill(7))

# Function to clean P_ADDRESS1
def clean_address(row):
    address = row['P_ADDRESS1']
    account_number = str(row['ACCOUNT_NUMBER'])
    if not isinstance(address, str) or address.strip() == '':
        if account_number.startswith('1'):
            address = 'addis abeba'
        elif account_number.startswith('2'):
            address = 'snnpr'
        elif account_number.startswith('3'):
            address = 'oromia'
    else:
        # Remove numerical values
        address = re.sub(r'\d+', '', address)
        # Standardize common misspellings
        address = address.strip().lower()
        address_corrections = {
            # Add address corrections here as required
        }
        for wrong, correct in address_corrections.items():
            if wrong in address:
                address = correct
    return address.capitalize()

# Apply the cleaning function to P_ADDRESS1
df['P_ADDRESS1'] = df.apply(clean_address, axis=1)

# Calculate Loan Tenure
df['LOAN_TENURE'] = (df['MATURITY_DATE'] - df['BOOK_DATE']).dt.days

# Calculate Payment Delays
df['PAYMENT_DELAY'] = (df['PAID_DATE'] - df['DUE_DATE']).dt.days
df['PAYMENT_DELAY'] = df['PAYMENT_DELAY'].apply(lambda x: max(x, 0))

# Identify Defaulted Loans based on STATUSDESCRIPTION
df['DEFAULTED'] = df.apply(lambda row: 1 if row['STATUSDESCRIPTION'] == 'Loss' else 0, axis=1)

# Additional logic to identify defaulted loans based on status descriptions
def determine_default(row):
    if row['STATUSDESCRIPTION'] == 'Loss':
        return 1
    elif row['STATUSDESCRIPTION'] in ['Doubtful', 'Substandard', 'Special Mention'] and row['PAID_DATE'] > row['MATURITY_DATE']:
        return 1
    return 0

df['DEFAULTED'] = df.apply(determine_default, axis=1)

# Add COMPONENT_NAME if not already in the dataset
if 'COMPONENT_NAME' not in df.columns:
    df['COMPONENT_NAME'] = ''

# Fill NaN values in COMPONENT_NAME with empty string to avoid errors in mask
df['COMPONENT_NAME'] = df['COMPONENT_NAME'].fillna('')

# Calculate Total Interest Paid
interest_mask = df['COMPONENT_NAME'].str.contains('INT')
df['TOTAL_INTEREST_PAID'] = df[interest_mask]['AMOUNT_PAID']

# Count PENALTY occurrences
penalty_counts = df[df['COMPONENT_NAME'].str.contains('PENALTY')].groupby('CUSTOMER_ID').size().reset_index(name='PENALTY_COUNT')

# Merge PENALTY_COUNT with the main dataframe
df = df.merge(penalty_counts, on='CUSTOMER_ID', how='left').fillna({'PENALTY_COUNT': 0})

# Identify customers with high PENALTY counts and categorize them as defaulted loans
df['DEFAULTED'] = df.apply(lambda row: 1 if row['PENALTY_COUNT'] > 3 else row['DEFAULTED'], axis=1) # Example threshold of 3 penalties

# Aggregations for Customer Loan Summary
customer_summary = df.groupby(['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1', 'STATUSDESCRIPTION', 'COMPONENT_NAME']).agg(
    Total_Amount_Financed=pd.NamedAgg(column='AMOUNT_FINANCED', aggfunc='sum'),
    Total_Amount_Repaid=pd.NamedAgg(column='AMOUNT_PAID', aggfunc='sum'),
    Number_of_Loans=pd.NamedAgg(column='ACCOUNT_NUMBER', aggfunc=pd.Series.nunique),
    Average_Loan_Tenure=pd.NamedAgg(column='LOAN_TENURE', aggfunc='mean'),
    Total_Payment_Delays=pd.NamedAgg(column='PAYMENT_DELAY', aggfunc='sum'),
    Total_Interest_Paid=pd.NamedAgg(column='TOTAL_INTEREST_PAID', aggfunc='sum'),
    Defaulted_Loans=pd.NamedAgg(column='DEFAULTED', aggfunc='sum'),
    PENALTY_COUNT=pd.NamedAgg(column='PENALTY_COUNT', aggfunc='max')
).reset_index()

# Calculate Principal vs. Interest Ratio
principal_mask = df['COMPONENT_NAME'].str.contains('PRINCIPAL')
interest_paid = df[interest_mask].groupby(['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1', 'STATUSDESCRIPTION', 'COMPONENT_NAME'])['AMOUNT_PAID'].sum().reset_index(name='Interest_Paid')
principal_paid = df[principal_mask].groupby(['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1', 'STATUSDESCRIPTION', 'COMPONENT_NAME'])['AMOUNT_PAID'].sum().reset_index(name='Principal_Paid')

principal_interest_ratio = pd.merge(principal_paid, interest_paid, on=['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1', 'STATUSDESCRIPTION', 'COMPONENT_NAME'], how='outer').fillna(0)
principal_interest_ratio['Principal_Interest_Ratio'] = principal_interest_ratio['Principal_Paid'] / principal_interest_ratio['Interest_Paid'].replace(0, 1)

# Merging the principal vs interest ratio to the customer summary
principal_interest_summary = principal_interest_ratio.groupby(['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1', 'STATUSDESCRIPTION', 'COMPONENT_NAME']).agg(
    Average_Principal_Interest_Ratio=pd.NamedAgg(column='Principal_Interest_Ratio', aggfunc='mean')
).reset_index()

customer_summary = pd.merge(customer_summary, principal_interest_summary, on=['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1', 'STATUSDESCRIPTION', 'COMPONENT_NAME'], how='left')

# Save the results to CSV files with error handling
output_file = 'customer_summary_risk_assessment.csv'
try:
    customer_summary.to_csv(output_file, index=False)
    print(f"File saved successfully as {output_file}")
except PermissionError:
    print(f"Permission denied: Unable to save the file as {output_file}.")
    alternative_file = 'customer_summary_risk_assessment_alternative.csv'
    customer_summary.to_csv(alternative_file, index=False)
    print(f"File saved successfully as {alternative_file} instead.")

# Load the preprocessed dataset
data = pd.read_csv('customer_summary_risk_assessment.csv')

# Define the features and target
X = data.drop('Defaulted_Loans', axis=1)
y = data['Defaulted_Loans']

# Identify categorical and numerical columns
categorical_cols = ['PRODUCT_CODE', 'STATUSDESCRIPTION', 'COMPONENT_NAME']
numerical_cols = ['Total_Payment_Delays', 'PENALTY_COUNT']

# Preprocessing pipeline for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply the preprocessing to the data
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

# Save the preprocessed data and the preprocessor
joblib.dump((X_train, X_test, y_train, y_test, preprocessor), 'preprocessed_data_with_transformer.pkl')
