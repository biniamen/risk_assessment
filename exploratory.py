import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the Matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Create the images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Load the dataset
data = pd.read_csv('customer_summary_risk_assessment.csv')

# Display the first few rows of the dataset
print(data.head())

# Basic statistics
print(data.describe())

# Setting the style for seaborn plots
sns.set(style="whitegrid")

# Function to format large numbers
def exact_amount(x, pos):
    'The two args are the value and tick position'
    return '%1.0f' % x

# Correcting duplicate locations with slightly different names
def clean_location(location):
    if pd.isna(location):
        return location
    location = location.strip().lower()
    corrections = {
        "addis ababa": ["addis ababa", "aa"],
        "snnprs": ["snnprs", "snnp", "southern nation nationality people", "southern nation nationality"],
        "oromia": ["oromia", "oro-region"],
        "dilla": ["dilla", "dilla-yerga cheffe"]
    }
    for correct, variants in corrections.items():
        if location in variants:
            return correct
    return location

data['P_ADDRESS1'] = data['P_ADDRESS1'].apply(clean_location)

# Filter locations with a non-zero total amount financed
filtered_data = data[data['Total_Amount_Financed'] > 0]

# 1. Distribution of Total Amount Financed
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['Total_Amount_Financed'], bins=50, kde=True)
plt.title('Histogram: Distribution of Total Amount Financed')
plt.xlabel('Total Amount Financed')
plt.ylabel('Frequency')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(exact_amount))
plt.tight_layout()
plt.savefig('images/distribution_total_amount_financed.png')
plt.close()

# 2. Distribution of Total Amount Repaid
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['Total_Amount_Repaid'], bins=50, kde=True)
plt.title('Histogram: Distribution of Total Amount Repaid')
plt.xlabel('Total Amount Repaid')
plt.ylabel('Frequency')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(exact_amount))
plt.tight_layout()
plt.savefig('images/distribution_total_amount_repaid.png')
plt.close()

# 3. Total Payment Delays by Product Code
plt.figure(figsize=(10, 6))
sns.boxplot(x='PRODUCT_CODE', y='Total_Payment_Delays', data=filtered_data)
plt.title('Boxplot: Total Payment Delays by Product Code')
plt.xlabel('Product Code')
plt.ylabel('Total Payment Delays')
plt.tight_layout()
plt.savefig('images/total_payment_delays_by_product_code.png')
plt.close()

# 4. Total Amount Financed by Location (P_ADDRESS1)
plt.figure(figsize=(14, 8))
location_order = filtered_data.groupby('P_ADDRESS1')['Total_Amount_Financed'].sum().sort_values(ascending=False).index
sns.barplot(x='P_ADDRESS1', y='Total_Amount_Financed', data=filtered_data, errorbar=None, order=location_order)
plt.title('Barplot: Total Amount Financed by Location')
plt.xlabel('Location')
plt.ylabel('Total Amount Financed')
plt.xticks(rotation=90)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(exact_amount))
plt.tight_layout()
plt.savefig('images/total_amount_financed_by_location.png')
plt.close()

# 5. Correlation Heatmap
plt.figure(figsize=(12, 10))
numeric_data = filtered_data.select_dtypes(include=[np.number])  # Select only numeric columns
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap: Correlation Heatmap')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png')
plt.close()

# 6. Scatter plot for Total Amount Financed vs. Total Amount Repaid
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total_Amount_Financed', y='Total_Amount_Repaid', hue='P_ADDRESS1', palette='tab10', data=filtered_data)
plt.title('Scatter Plot: Total Amount Financed vs. Total Amount Repaid')
plt.xlabel('Total Amount Financed')
plt.ylabel('Total Amount Repaid')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(exact_amount))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(exact_amount))
plt.tight_layout()
plt.savefig('images/total_amount_financed_vs_total_amount_repaid.png')
plt.close()

# 7. Distribution of Loan Tenure
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['Average_Loan_Tenure'], bins=50, kde=True)
plt.title('Histogram: Distribution of Average Loan Tenure')
plt.xlabel('Average Loan Tenure (days)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('images/distribution_average_loan_tenure.png')
plt.close()

# 8. Donut Chart for Product Code Distribution
product_code_counts = filtered_data['PRODUCT_CODE'].value_counts()
plt.figure(figsize=(10, 8))

# Create the donut chart without autopct
wedges, texts = plt.pie(
    product_code_counts,
    labels=None,
    startangle=140,
    wedgeprops=dict(width=0.3)
)

# Add a legend with labels
plt.legend(
    wedges, [f'{label} ({p:.1f}%)' for label, p in zip(product_code_counts.index, product_code_counts / product_code_counts.sum() * 100)],
    title="Product Code",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)

# Add a central circle to make it a donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Title
plt.title('Donut Chart: Distribution of Loans by Product Code')
plt.tight_layout()
plt.savefig('images/donut_chart_product_code_distribution.png')
plt.close()
