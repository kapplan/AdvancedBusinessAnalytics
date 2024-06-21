import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import shap

# Load the dataset
file_path = '/Users/apple/Downloads/claims_q12023.csv'  # Update this to your file's path
data = pd.read_csv(file_path, delimiter=';', header=0)

print(data.head())

column_data_types = data.dtypes
print(column_data_types)
print(data.dtypes)

# Unique values to check for the number of categories
unique_values = {col: data[col].unique() for col in data.columns}
print(unique_values)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Generate descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# Replace '?' with np.nan in the 'property_damage' column, and 'None' in 'emg_services_notified' column with np.nan, and empty cells with nan
data['property_damage'] = data['property_damage'].replace('?', np.nan)
data['emg_services_notified'] = data['emg_services_notified'].replace('None', np.nan)
data['police_report_avlbl'] = data['police_report_avlbl'].replace('', np.nan)

# Visualizing the missingness pattern
msno.matrix(data)
plt.show()

# Heatmap to show correlations of missingness between different columns
msno.heatmap(data)
plt.show()

# Bar chart to show the count of missing values in each column
msno.bar(data)
plt.show()

# Cross Tabulation
for col in ['cust_region', 'sum_assured_group', 'insured_sex', 'edu_lvl', 'marital_status', 'claim_type']:
    print(pd.crosstab(data[col], data['property_damage'].isnull()))
    print(pd.crosstab(data[col], data['emg_services_notified'].isnull()))
    print(pd.crosstab(data[col], data['police_report_avlbl'].isnull()))

# We assume that for 'parked car' and 'theft' incidents, the 'acc_type' should be set to 'Not Applicable' and 'Minor incident',
# because the nature of missingness indicate that there is
# Set 'acc_type' to 'Not Applicable' for 'theft' incidents
data.loc[data['claim_type'] == 'theft', 'acc_type'] = 'Not Applicable'
# Set 'acc_type' to 'Minor incident' for 'parked car' incidents
data.loc[data['claim_type'] == 'parked car', 'acc_type'] = 'Minor incident'

# Segment 'cust_age' into age groups
data['Age_Group'] = pd.cut(data['cust_age'], bins=[18, 25, 35, 45, 55, 65, np.inf],
                           labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
# Drop the 'cust_age' column if you decide it's no longer needed
data.drop('cust_age', axis=1, inplace=True)

# Feature Engineering for risk analysis: Claim Frequency and Severity Metrics
data['claim_amount_per_vehicle'] = data['total_claim_amount'] / data['num_vehicles_involved']
print(data)

# Create an imputer object with a strategy of replacing missing values with the most frequent value
mode_imputer = SimpleImputer(strategy='most_frequent')
# List of columns to be imputed
columns_to_impute = ['property_damage', 'police_report_avlbl']
# Apply the imputer to the selected columns
for column in columns_to_impute:
    # The imputer returns a 2D array, so we need to select the first (and only) column of this array
    data[column] = mode_imputer.fit_transform(data[[column]]).ravel()

# Display the head of the DataFrame to check the changes
print(data.head())
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Replace 'numerical_cols' and 'categorical_cols' with your column names
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Create transformers for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply the preprocessor
X_preprocessed = preprocessor.fit_transform(data)
from sklearn.cluster import KMeans

# Elbow methods show 5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X_preprocessed)
import matplotlib.pyplot as plt

inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, init='k-means++', random_state=42)
    kmeans.fit(X_preprocessed)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# One hot encoding columns, PCA and k-meanss clustering
# Define which columns are categorical
categorical_cols = data.select_dtypes(include=['object', 'bool']).columns.tolist()
# Define which columns are numerical
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
# Applying the preprocessor
X_preprocessed = preprocessor.fit_transform(data)

# TruncatedSVD with a high number of components
from sklearn.decomposition import TruncatedSVD

# Calculate the cumulative explained variance ratio
cumulative_variance = np.cumsum(svd.explained_variance_ratio_)

# Determine the number of components to retain 95% of the variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1  # +1 because indices start at 0

# Now, reapply TruncatedSVD with the chosen number of components
svd_95 = TruncatedSVD(n_components=n_components_95)
X_svd_95 = svd_95.fit_transform(X_preprocessed)

# Then determine the optimal number of clusters using silhouette scores
range_n_clusters = range(2, 11)
best_n_clusters = 0
best_silhouette_score = -1

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X_svd)  # Using SVD-transformed data
    silhouette_avg = silhouette_score(X_svd, cluster_labels)

    # Check if this silhouette score is the best one so far
    if silhouette_avg > best_silhouette_score:
        best_n_clusters = n_clusters
        best_silhouette_score = silhouette_avg

# Now that you have the optimal number of clusters, apply K-Means clustering to the chosen data representation
kmeans = KMeans(n_clusters=best_n_clusters, init='k-means++', random_state=42)
# Use X_svd if you decided to use SVD-transformed data for final clustering
cluster_labels = kmeans.fit_predict(X_svd)

# Create a DataFrame for the SVD results for easy plotting
column_names = [f'Component_{i+1}' for i in range(X_svd.shape[1])]
svd_df = pd.DataFrame(X_svd, columns=column_names)
svd_df['Cluster'] = cluster_labels

# Plotting the clusters using the first two SVD components
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Component_1', y='Component_2', hue='Cluster', data=svd_df, palette='viridis')
plt.title('Cluster Visualization with TruncatedSVD Components')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.show()

# Step 4: Final K-means Clustering
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
data['segment_label'] = kmeans_final.fit_predict(features_pca)

# Step 5: Visualization of Segments
plt.figure(figsize=(10, 6))
sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=data['segment_label'], palette='viridis')
plt.title('Customer Segments after PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Segment')
plt.show()

# Additional segment analysis and visualizations
# Segment-wise Descriptive Statistics
for i in range(optimal_clusters):
    print(f"\nSegment {i} Statistics:\n", data[data['segment_label'] == i].describe())

# Visualizing Segment-wise Average Claim Amounts
plt.figure(figsize=(10, 6))
sns.barplot(x='segment_label', y='total_claim_amount', data=data, estimator=np.mean)
plt.title('Average Total Claim Amount by Segment')
plt.xlabel('Segment')
plt.ylabel('Average Total Claim Amount')
plt.show()

# Visualizing Segment-wise Customer Age Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x='segment_label', y='age_group', data=data)
plt.title('Customer Age Group Distribution by Segment')
plt.xlabel('Segment')
plt.ylabel('Customer Age Group')
plt.show()

# Visualizing Segment-wise Annual Premium
plt.figure(figsize=(10, 6))
sns.barplot(x='segment_label', y='annual_prem', data=data, estimator=np.mean)
plt.title('Average Annual Premium by Segment')
plt.xlabel('Segment')
plt.ylabel('Average Annual Premium')
plt.show()