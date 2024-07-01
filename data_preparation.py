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
                'snnprs': 'snnpr',
                'misrak': 'eastern',
                'sidama region': 'sidama',
                'alamura': 'hawassa',
                'sidama': 'sidama',
                'oro-region': 'oromia',
                'A-a' : 'addis abeba',
                'snnpr': 'snnpr',
                'bole': 'addis abeba',
                'kolfe keranyo': 'addis abeba',
                'southern region': 'snnpr',
                'yirgacheffe': 'sidama',
                'kirkos k/ketema': 'addis abeba',
                'yeka': 'addis abeba',
                'arada': 'addis abeba',
                'dilla': 'sidama',
                'gedeb': 'gamo',
                'ktz': 'sidama',
                'durame': 'kembata',
                'chuko': 'sidama',
                '147a-a': 'addis abeba',
                'hawassa': 'hawassa',
                'dukem': 'oromia',
                'nifas silk lafto': 'addis abeba',
                'dilla city': 'sidama',
                'bahel adarash': 'addis abeba',
                'meneharia': 'addis abeba',
                'south  region': 'snnpr',
                's/n/n/p': 'snnpr',
                'k/kerno sub city': 'addis abeba',
                'oro': 'oromia',
                'werabe': 'gurage',
                'oromia region': 'oromia',
                'worabe': 'gurage',
                'oromia': 'oromia',
                'k/keranio': 'addis abeba',
                'oromia zone burayu': 'oromia',
                'kirkos': 'addis abeba',
                'addis ababa': 'addis abeba',
                'amaro kele': 'gamo',
                'n/lafto': 'addis abeba',
                'bole': 'addis abeba',
                'alelu': 'oromia',
                'amhara': 'amhara',
                'aa': 'addis abeba',
                'south nation nationality people reg': 'snnpr',
                'shashemene': 'oromia',
                'adama': 'oromia',
                'chelelekitu': 'sidama',
                'y/cheffe': 'sidama',
                'darra': 'sidama',
                'tabor': 'sidama',
                'daye': 'sidama',
                'fisehagenet': 'addis abeba',
                'menharia': 'addis abeba',
                'kolfe': 'addis abeba',
                'south nation nationality people': 'snnpr',
                'a/chuko': 'sidama',
                'dilla ketema': 'sidama',
                'tigray': 'tigray',
                'akike kality': 'addis abeba',
                'bole w 14': 'addis abeba',
                'bekur': 'addis abeba',
                'n/s/lafto': 'addis abeba',
                'southern': 'snnpr',
                'southern nation nationality people': 'snnpr',
                'addis ketema': 'addis abeba',
                'gulele': 'addis abeba',
                'k.k': 'addis abeba',
                'l/s/lafto': 'addis abeba',
                'chicko': 'sidama',
                'addis abab': 'addis abeba',
                'lideta': 'addis abeba',
                'gullele': 'addis abeba',
                'lidita': 'addis abeba',
                'kolfe keraneyo': 'addis abeba',
                'lafto': 'addis abeba',
                'ledeta': 'addis abeba',
                'a/ketema': 'addis abeba',
                'diredawa': 'dire dawa',
                'a.a': 'addis abeba',
                'n/lafto': 'addis abeba',
                'akaki kality': 'addis abeba',
                'n/l/k': 'addis abeba',
                'n/s/l': 'addis abeba',
                'aa': 'addis abeba',
                'merkato': 'addis abeba',
                'a/a': 'addis abeba',
                's/s/lafto sub city': 'addis abeba',
                'bole sub city': 'addis abeba',
                'n/l': 'addis abeba',
                'doyogena': 'kembata',
                'a/ketema w10': 'addis abeba',
                'south region': 'snnpr',
                'kolfe keranio': 'addis abeba',
                'a/kality': 'addis abeba',
                'k/k': 'addis abeba',
                'aa lideta': 'addis abeba',
                'yika': 'addis abeba',
                'h/wolabo': 'oromia',
                'aletachuko': 'sidama',
                'k/keraneyon': 'addis abeba',
                'lidea': 'addis abeba',
                'd/genna': 'gurage',
                'sech duna': 'gurage',
                'g/meda': 'addis abeba',
                'hossana': 'hadiya',
                'south nation nationality and people': 'snnpr',
                'kerkos': 'addis abeba',
                'hosana': 'hadiya',
                'betel': 'addis abeba',
                'nifas silk/lafto': 'addis abeba',
                'gofer meda': 'addis abeba',
                'southern nation nationality': 'snnpr',
                'gullel': 'addis abeba',
                'adea': 'oromia',
                'staff': 'addis abeba',
                'yeke': 'addis abeba',
                'gulelle': 'addis abeba',
                'sodo': 'wolaita',
                'bodity': 'wolaita',
                'adisabeba': 'addis abeba',
                'boditi': 'wolaita',
                'hawasa': 'hawassa',
                'awassa': 'hawassa',
                'addis abeba': 'addis abeba',
                'aleta chuko': 'sidama',
                'mekele': 'tigray',
                'yerga alem': 'sidama',
                'menharia': 'addis abeba',
                'gofermeda': 'addis abeba',
                'oro. region': 'oromia',
                'oromia spetial zone': 'oromia',
                'dila': 'sidama',
                'chuko 01': 'sidama',
                'mennaharia': 'addis abeba'
        }
        for wrong, correct in address_corrections.items():
            if wrong in address:
                address = correct
    return address.capitalize()

# Apply the cleaning function to P_ADDRESS1
df['P_ADDRESS1'] = df.apply(clean_address, axis=1)

# Calculate Loan Tenure
df['LOAN_TENURE'] = (df['MATURITY_DATE'] - df['BOOK_DATE']).dt.days

# Calculate Age at Loan Booking
#df['AGE_AT_LOAN'] = (df['BOOK_DATE'] - df['DATE_OF_BIRTH']).dt.days // 365

# Calculate Payment Delays
df['PAYMENT_DELAY'] = (df['PAID_DATE'] - df['DUE_DATE']).dt.days
df['PAYMENT_DELAY'] = df['PAYMENT_DELAY'].apply(lambda x: max(x, 0))

# Identify Defaulted Loans
df['DEFAULTED'] = df.apply(lambda row: row['PAID_DATE'] > row['MATURITY_DATE'], axis=1)

# Fill NaN values in COMPONENT_NAME with empty string to avoid errors in mask
df['COMPONENT_NAME'] = df['COMPONENT_NAME'].fillna('')

# Calculate Total Interest Paid
interest_mask = df['COMPONENT_NAME'].str.contains('INT')
df['TOTAL_INTEREST_PAID'] = df[interest_mask]['AMOUNT_PAID']

# Aggregations for Customer Loan Summary
customer_summary = df.groupby(['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1']).agg(
    Total_Amount_Financed=pd.NamedAgg(column='AMOUNT_FINANCED', aggfunc='sum'),
    Total_Amount_Repaid=pd.NamedAgg(column='AMOUNT_PAID', aggfunc='sum'),
    Number_of_Loans=pd.NamedAgg(column='ACCOUNT_NUMBER', aggfunc=pd.Series.nunique),
    Average_Loan_Tenure=pd.NamedAgg(column='LOAN_TENURE', aggfunc='mean'),
    #Age_at_Loan=pd.NamedAgg(column='AGE_AT_LOAN', aggfunc='mean'),
    Total_Payment_Delays=pd.NamedAgg(column='PAYMENT_DELAY', aggfunc='sum'),
    Total_Interest_Paid=pd.NamedAgg(column='TOTAL_INTEREST_PAID', aggfunc='sum'),
    Defaulted_Loans=pd.NamedAgg(column='DEFAULTED', aggfunc='sum')
).reset_index()

# Calculate Principal vs. Interest Ratio
principal_mask = df['COMPONENT_NAME'].str.contains('PRINCIPAL')
interest_paid = df[interest_mask].groupby(['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1'])['AMOUNT_PAID'].sum().reset_index(name='Interest_Paid')
principal_paid = df[principal_mask].groupby(['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1'])['AMOUNT_PAID'].sum().reset_index(name='Principal_Paid')

principal_interest_ratio = pd.merge(principal_paid, interest_paid, on=['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1'], how='outer').fillna(0)
principal_interest_ratio['Principal_Interest_Ratio'] = principal_interest_ratio['Principal_Paid'] / principal_interest_ratio['Interest_Paid'].replace(0, 1)

# Merging the principal vs interest ratio to the customer summary
principal_interest_summary = principal_interest_ratio.groupby(['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1']).agg(
    Average_Principal_Interest_Ratio=pd.NamedAgg(column='Principal_Interest_Ratio', aggfunc='mean')
).reset_index()

customer_summary = pd.merge(customer_summary, principal_interest_summary, on=['CUSTOMER_ID', 'ACCOUNT_NUMBER', 'PRODUCT_CODE', 'P_ADDRESS1'], how='left')

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
categorical_cols = ['PRODUCT_CODE', 'P_ADDRESS1']
numerical_cols = ['Total_Amount_Financed', 'Total_Amount_Repaid', 'Number_of_Loans',
                  'Average_Loan_Tenure', 'Total_Payment_Delays',
                  'Total_Interest_Paid', 'Average_Principal_Interest_Ratio']

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
