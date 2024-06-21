import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

# Replace empty strings with np.nan
data.replace('', np.nan, inplace=True)

# Replace '?' with np.nan in the 'property_damage' column, and 'None' in 'emg_services_notified' column with np.nan
data['property_damage'] = data['property_damage'].replace('?', np.nan)
data['emg_services_notified'] = data['emg_services_notified'].replace('None', np.nan)

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

# We assume that for 'parked car' and 'theft' incidents, the 'acc_type' should be set to 'Not Applicable' and 'Minor incident'
# Set 'acc_type' to 'Not Applicable' for 'theft' incidents
data.loc[data['claim_type'] == 'theft', 'acc_type'] = 'Not Applicable'

# Set 'acc_type' to 'Minor incident' for 'parked car' incidents
data.loc[data['claim_type'] == 'parked car', 'acc_type'] = 'Minor incident'

# Segment 'cust_age' into age groups, store them in a new column
data['Age_Group'] = pd.cut(data['cust_age'], bins=[18, 25, 35, 45, 55, 65, np.inf], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
# Drop the 'cust_age' column if you decide it's no longer needed
data.drop('cust_age', axis=1, inplace=True)

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


# You can start with a random guess for the number of clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X_preprocessed)

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

# Optimal clusters from elbow method
optimal_clusters = 4
kmeans_optimal = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
data['cluster'] = kmeans_optimal.fit_predict(X_preprocessed)

# Analyze the clusters
for i in range(optimal_clusters):
    print(f"Cluster {i} Summary:")
    print(data[data['cluster'] == i].describe())

from sklearn.metrics import silhouette_score

# Assuming 'X_preprocessed' is your feature set and 'clusters' is the array of cluster labels
score = silhouette_score(X_preprocessed, data['cluster'])
print('Silhouette Score:', score)

#One hot encoding columns, PCA and k-meanss clustering

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

# Apply PCA to reduce dimensionality
pca = PCA()
X_pca = pca.fit_transform(X_preprocessed)

# Choose the optimal number of components based on explained variance
n_components = next(x[0] for x in enumerate(np.cumsum(pca.explained_variance_ratio_)) if x[1] > 0.95) + 1
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_preprocessed)

# Apply K-Means clustering on the PCA-reduced data
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
data['cluster'] = kmeans.fit_predict(X_pca)

# Calculate silhouette score
score = silhouette_score(X_pca, data['cluster'])
print('Silhouette Score:', score)
