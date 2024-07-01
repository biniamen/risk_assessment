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

# Filter locations with a non-zero total amount financed
filtered_data = data[data['Total_Amount_Financed'] > 0]

# 1. Distribution of Penalty Count
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['PENALTY_COUNT'], bins=50, kde=True)
plt.title('Histogram: Distribution of Penalty Count')
plt.xlabel('Penalty Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('images/distribution_penalty_count.png')
plt.close()

# 2. Distribution of Total Payment Delays
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['Total_Payment_Delays'], bins=50, kde=True)
plt.title('Histogram: Distribution of Total Payment Delays')
plt.xlabel('Total Payment Delays')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('images/distribution_total_payment_delays.png')
plt.close()

# 3. Total Payment Delays by Status Description
plt.figure(figsize=(10, 6))
sns.boxplot(x='STATUSDESCRIPTION', y='Total_Payment_Delays', data=filtered_data)
plt.title('Boxplot: Total Payment Delays by Status Description')
plt.xlabel('Status Description')
plt.ylabel('Total Payment Delays')
plt.tight_layout()
plt.savefig('images/total_payment_delays_by_status_description.png')
plt.close()

# 4. Penalty Count by Status Description
plt.figure(figsize=(10, 6))
sns.boxplot(x='STATUSDESCRIPTION', y='PENALTY_COUNT', data=filtered_data)
plt.title('Boxplot: Penalty Count by Status Description')
plt.xlabel('Status Description')
plt.ylabel('Penalty Count')
plt.tight_layout()
plt.savefig('images/penalty_count_by_status_description.png')
plt.close()

# 5. Correlation Heatmap for Critical Columns
plt.figure(figsize=(12, 10))
data_encoded = pd.get_dummies(filtered_data, columns=['STATUSDESCRIPTION', 'COMPONENT_NAME'], drop_first=True)
columns_of_interest = ['Total_Payment_Delays', 'PENALTY_COUNT'] + \
                      [col for col in data_encoded.columns if 'STATUSDESCRIPTION' in col or 'COMPONENT_NAME' in col]
correlation_matrix = data_encoded[columns_of_interest].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap: Correlation of Critical Columns with STATUSDESCRIPTION and COMPONENT_NAME')
plt.tight_layout()
plt.savefig('images/correlation_heatmap_updated.png')
plt.close()

# 6. Scatter plot for Total Payment Delays vs. Penalty Count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total_Payment_Delays', y='PENALTY_COUNT', hue='STATUSDESCRIPTION', palette='tab10', data=filtered_data)
plt.title('Scatter Plot: Total Payment Delays vs. Penalty Count')
plt.xlabel('Total Payment Delays')
plt.ylabel('Penalty Count')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.savefig('images/total_payment_delays_vs_penalty_count.png')
plt.close()

# 7. Donut Chart for Status Description Distribution
status_description_counts = filtered_data['STATUSDESCRIPTION'].value_counts()
plt.figure(figsize=(10, 8))

# Create the donut chart without autopct
wedges, texts = plt.pie(
    status_description_counts,
    labels=None,
    startangle=140,
    wedgeprops=dict(width=0.3)
)

# Add a legend with labels
plt.legend(
    wedges, [f'{label} ({p:.1f}%)' for label, p in zip(status_description_counts.index, status_description_counts / status_description_counts.sum() * 100)],
    title="Status Description",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)

# Add a central circle to make it a donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Title
plt.title('Donut Chart: Distribution of Loans by Status Description')
plt.tight_layout()
plt.savefig('images/donut_chart_status_description_distribution.png')
plt.close()
