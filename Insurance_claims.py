# Import libraries
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#%%
data = pd.read_csv('/Users/apple/Downloads/claims_q12023.csv', delimiter=';')
#%%
# Initial EDA: Check for missing values

# Print the columns
columns = data.dtypes
print(columns)

rows, columns = data.shape

print(f'The dataset contains {rows} rows and {columns} columns.')

# Replace '?' with np.nan in the 'property_damage' column, 'None' in 'emg_services_notified' column with np.nan,
# empty spaces with np.nan too.
data['property_damage'] = data['property_damage'].replace('?', np.nan)
data['emg_services_notified'] = data['emg_services_notified'].replace('None', np.nan)
data['police_report_avlbl'] = data['police_report_avlbl'].replace('', np.nan)

# Generate descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# Convert dates from strings to datetime objects with the correct format
data['coverage_start_date'] = pd.to_datetime(data['coverage_start_date'], format='%d.%m.%Y')
data['claim_incurred_date'] = pd.to_datetime(data['claim_incurred_date'], format='%d.%m.%Y')

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
#%%
# We assume that for 'parked car' and 'theft' incidents, the 'acc_type' should be set to 'Not Applicable' and 'Minor incident',
# because the nature of missingness indicate that there is
# Set 'acc_type' to 'Not Applicable' for 'theft' incidents
data.loc[data['claim_type'] == 'theft', 'acc_type'] = 'Not Applicable'

# Set 'acc_type' to 'Minor incident' for 'parked car' incidents
data.loc[data['claim_type'] == 'parked car', 'acc_type'] = 'Minor incident'

# Impute missing values with the most common value for each of the three columns
for column in ['emg_services_notified', 'property_damage', 'police_report_avlbl']:
    most_common_value = data[column].mode()[0]
    data[column].fillna(most_common_value, inplace=True)
    print(f"Imputed missing values in '{column}' with '{most_common_value}'.")

# Check missing values count again to ensure they are filled
print("\nMissing values per column after imputation:")
print(data.isnull().sum())
#%%
# Segment 'cust_age' into age groups
data['age_group'] = pd.cut(data['cust_age'], bins=[18, 25, 35, 45, 55, 65, np.inf], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])

# New features for risk analysis: Claim Frequency and Severity Metrics
# Claim Frequency Features
# Count total number of claims by policy_id
data['total_claims'] = data.groupby('policy_id')['policy_id'].transform('count')

# Claim Severity Features
# Average, max, and total claim amount by policy_id
data['avg_claim_amount'] = data.groupby('policy_id')['total_claim_amount'].transform('mean')
data['max_claim_amount'] = data.groupby('policy_id')['total_claim_amount'].transform('max')
data['total_claimed_amount'] = data.groupby('policy_id')['total_claim_amount'].transform('sum')

# Net Contribution: The net contribution of a customer to the company is the difference between the premiums paid and the claims cost. A positive value indicates profitability, while a negative value indicates a loss
# Calculate the duration of each policy in years
data['policy_duration_years'] = (data['claim_incurred_date'] - data['coverage_start_date']).dt.days / 365.25

# Calculate the total premiums paid by each customer
# Assuming 'annual_prem' is the annual premium
data['total_premiums_paid'] = data['annual_prem'] * data['policy_duration_years']

# Calculate the total claims cost for each policy
# Assuming 'total_claim_amount' is the claim amount per incident
data['total_claims_cost'] = data.groupby('policy_id')['total_claim_amount'].transform('sum')

# Calculate the net contribution for each policy
data['net_contribution'] = data['total_premiums_paid'] - data['total_claims_cost']
#%%
# Data Visualization
# Histogram of Customer Ages
plt.figure(figsize=(10, 6))
sns.histplot(data['age_group'], bins=30, kde=True)
plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.show()

#Average Claim amount by age
plt.figure(figsize=(10, 6))
sns.barplot(x='age_group', y='avg_claim_amount', data=data)
plt.title('Average Claim Amount by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Claim Amount')
plt.show()

# Boxplot of Total Claim Amount by Customer Region
plt.figure(figsize=(10, 6))
sns.boxplot(x='cust_region', y='total_claimed_amount', data=data)
plt.title('Total Claim Amount by Customer Region')
plt.xlabel('Customer Region')
plt.ylabel('Total Claim Amount')
plt.show()
#%%
# Checking the distribution of key features
data['time_to_claim'] = (data['claim_incurred_date'] - data['coverage_start_date']).dt.days

for col in ['annual_prem', 'total_claim_amount', 'time_to_claim']:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.ylabel('Frequency')
    plt.xlabel(col)
    plt.show()

# Checking for outliers
for col in ['annual_prem', 'total_claim_amount']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

#%%
# Profitability
sns.histplot(data['net_contribution'], bins=20, kde=True)
plt.title('Net Contribution Distribution')
plt.xlabel('Net Contribution')
plt.ylabel('Frequency')
plt.show()

sns.scatterplot(x='total_premiums_paid', y='net_contribution', data=data)
plt.title('Net Contribution vs Total Premiums Paid')
plt.xlabel('Total Premiums Paid')
plt.ylabel('Net Contribution')
plt.show()

# Identify the top 10 insurance IDs with the most losses
most_losses = data.nsmallest(20, 'net_contribution')

# Print the policy IDs with the most losses
print("Policy IDs with the most losses:\n", most_losses[['policy_id', 'net_contribution', 'car_brand','car_model']].to_string(index=False))

# Additional analysis: Summary statistics for high-loss policies
print("\nSummary Statistics for Policies with the Least Profits:\n", most_losses.describe())

#%%
# First Approach to Clustering - Silhouette and Elbow Method for K-Means
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

# Define the range of clusters to try
range_n_clusters = range(2, 11)

# Initialize lists to store the results of the silhouette score and inertia
silhouette_avg_scores = []
inertia_scores = []

# Calculate silhouette score and inertia for different numbers of clusters
for n_clusters in range_n_clusters:
    # Initialize the clusterer with n_clusters value and a random generator seed for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X_preprocessed)

    # Silhouette score
    silhouette_avg = silhouette_score(X_preprocessed, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)

    # Inertia
    inertia_scores.append(clusterer.inertia_)

# Plotting the silhouette scores
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(range_n_clusters, silhouette_avg_scores, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal Number of Clusters')

# Plotting the elbow method results
plt.subplot(1, 2, 2)
plt.plot(range_n_clusters, inertia_scores, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method For Optimal Number of Clusters')

plt.tight_layout()
plt.show()
#%%
# Choose the number of clusters based on the plots and set it in the final K-means model
optimal_clusters = 3

# List of numeric columns
numeric_columns = [
    'cust_age', 'ins_deductible', 'annual_prem', 'zip_code',
    'incident_hour', 'num_vehicles_involved', 'bodily_injuries',
    'witnesses', 'injury_claim', 'property_claim', 'vehicle_claim',
    'production_year'
]

# Ensure numeric columns are correctly typed
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# List of categorical columns to convert
categorical_cols = [
    'policy_id', 'coverage_start_date', 'cust_region', 'sum_assured_group',
    'insured_sex', 'edu_lvl', 'marital_status', 'claim_incurred_date',
    'claim_type', 'acc_type', 'emg_services_notified', 'incident_city',
    'property_damage', 'police_report_avlbl', 'car_brand', 'car_model'
]

# Converting categorical features to numerical using pd.get_dummies()
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Extract features for clustering
dummy_cols = data.columns[data.columns.str.startswith(tuple(categorical_cols))]
features = data[numeric_columns + list(dummy_cols)]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(features_scaled)

# Analyze segment characteristics - only numeric columns
numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
segment_summary = data.groupby('cluster')[numeric_cols].mean()
print("\nSegment Characteristics:\n", segment_summary)

# Visualize the clusters using the first two PCA components
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=data['cluster'], palette='viridis')
plt.title('Customer Segments Based on Profitability')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend(title='Cluster')
plt.show()

#%%
# Select numerical and categorical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'bool']).columns.tolist()

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

# Apply the preprocessor
X_preprocessed = preprocessor.fit_transform(data)

# Define the range of clusters to try
range_n_clusters = range(2, 11)

# Initialize lists to store the results of the silhouette score and inertia
silhouette_avg_scores = []
inertia_scores = []

# Calculate silhouette score and inertia for different numbers of clusters
for n_clusters in range_n_clusters:
    # Initialize the clusterer with n_clusters value and a random generator seed for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X_preprocessed)

    # Silhouette score
    silhouette_avg = silhouette_score(X_preprocessed, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)

    # Inertia
    inertia_scores.append(clusterer.inertia_)

# Plotting the silhouette scores
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(range_n_clusters, silhouette_avg_scores, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal Number of Clusters')

# Plotting the elbow method results
plt.subplot(1, 2, 2)
plt.plot(range_n_clusters, inertia_scores, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method For Optimal Number of Clusters')

plt.tight_layout()
plt.show()

#%%
# Second Approach - TruncatedSVD, K-Means clustering,
# Apply TruncatedSVD with a high number of components
svd = TruncatedSVD(n_components=50)
X_svd = svd.fit_transform(X_preprocessed)

# Calculate the cumulative explained variance ratio
cumulative_variance = np.cumsum(svd.explained_variance_ratio_)

# Determine the number of components to retain 95% of the variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1  # +1 because indices start at 0

# Now, reapply TruncatedSVD with the chosen number of components
svd_95 = TruncatedSVD(n_components=n_components_95)
X_svd_95 = svd_95.fit_transform(X_preprocessed)

# Determine the optimal number of clusters using silhouette scores
range_n_clusters = range(2, 11)
best_n_clusters = 0
best_silhouette_score = -1

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X_svd_95)  # Using SVD-transformed data
    silhouette_avg = silhouette_score(X_svd_95, cluster_labels)

    # Check if this silhouette score is the best one so far
    if silhouette_avg > best_silhouette_score:
        best_n_clusters = n_clusters
        best_silhouette_score = silhouette_avg

# Now that you have the optimal number of clusters, apply K-Means clustering to the chosen data representation
kmeans = KMeans(n_clusters=best_n_clusters, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(X_svd_95)

# Create a DataFrame for the SVD results for easy plotting
column_names = [f'Component_{i+1}' for i in range(X_svd_95.shape[1])]
svd_df = pd.DataFrame(X_svd_95, columns=column_names)
svd_df['Cluster'] = cluster_labels

# Plotting the clusters using the first two SVD components
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Component_1', y='Component_2', hue='Cluster', data=svd_df, palette='viridis')
plt.title('Cluster Visualization with TruncatedSVD Components')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.show()

# Isolation Forest
# Identify outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)  # contamination is an estimate of the proportion of outliers
outliers = iso_forest.fit_predict(data.select_dtypes(include=[np.number]))

# Find the indices of outliers
outlier_indices = np.where(outliers == -1)[0]

# Map the indices to original data to get suspicious policy IDs
suspicious_policy_ids = data.iloc[outlier_indices]['policy_id'].values

# Print suspicious policy IDs
print("Suspicious Policy IDs:\n", suspicious_policy_ids)

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svd', svd_95),
    ('model', LinearRegression())
])

# Fit the pipeline
model.fit(X, data['total_claim_amount'])

# Transform the input data for SHAP analysis
X_transformed = model.named_steps['preprocessor'].transform(X)

# Generate new feature names after one-hot encoding
new_feature_names = numerical_cols.copy()
for feature in categorical_cols:
    unique_values = data[feature].unique()
    for value in unique_values:
        new_feature_names.append(f"{feature}_{value}")
all_feature_names = new_feature_names

# Explain the model using SHAP
explainer = shap.Explainer(model.named_steps['model'], X_transformed)

# Calculate SHAP values for the transformed data
shap_values = explainer(X_transformed)

# Plot the summary of SHAP values to show feature importance
shap.summary_plot(shap_values, features=X_transformed, feature_names=all_feature_names)
#%%
# Further analysis of profitable and non-profitable segments
# Define profitable and not profitable based on net contribution
data['profitable'] = np.where(data['net_contribution'] > 0, 'Profitable', 'Not Profitable')

profitable_customers = data[data['net_contribution'] > 0]
non_profitable_customers = data[data['net_contribution'] < 0]

# Summary statistics for profitable customers
print("\nSummary Statistics for Profitable Customers:\n", profitable_customers.describe())

# Summary statistics for non-profitable customers
print("\nSummary Statistics for Non-Profitable Customers:\n", non_profitable_customers.describe())

# Visualize distribution of age groups in profitable vs non-profitable segments
plt.figure(figsize=(12, 6))
sns.countplot(x='cust_age', hue='profitable', data=data)
plt.title('Distribution of Age Groups in Profitable vs Non-Profitable Segments')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Profitability')
plt.show()

# Visualize distribution of claim types in profitable vs non-profitable segments
plt.figure(figsize=(12, 6))

# Extract claim type columns
claim_type_cols = [col for col in data.columns if col.startswith('claim_type_')]
claim_type_data = data.melt(id_vars=['profitable'], value_vars=claim_type_cols,
                            var_name='claim_type', value_name='value')
claim_type_data = claim_type_data[claim_type_data['value'] == 1]

sns.countplot(x='claim_type', hue='profitable', data=claim_type_data)
plt.title('Distribution of Claim Types in Profitable vs Non-Profitable Segments')
plt.xlabel('Claim Type')
plt.ylabel('Count')
plt.legend(title='Profitability')
plt.xticks(rotation=45)
plt.show()

# Visualize distribution of annual premiums in profitable vs non-profitable segments
plt.figure(figsize=(12, 6))
sns.boxplot(x='profitable', y='annual_prem', data=data)
plt.title('Distribution of Annual Premiums in Profitable vs Non-Profitable Segments')
plt.xlabel('Profitability')
plt.ylabel('Annual Premium')
plt.show()

# Exclude non-numeric columns for correlation analysis
profitable_numeric = profitable_customers.select_dtypes(include=[np.number])
non_profitable_numeric = non_profitable_customers.select_dtypes(include=[np.number])

# Correlation analysis in profitable vs non-profitable segments
profitable_corr = profitable_numeric.corr()
non_profitable_corr = non_profitable_numeric.corr()

plt.figure(figsize=(12, 6))
sns.heatmap(profitable_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Profitable Customers')
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(non_profitable_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Non-Profitable Customers')
plt.show()

# Summary of the findings and strategies
print("\nSummary of Findings and Strategies:\n")
print(f"Total Net Loss: {data[data['net_contribution'] < 0]['net_contribution'].sum()}")
print(f"Total Net Profit: {data[data['net_contribution'] > 0]['net_contribution'].sum()}")

print("\nProfitable Customer Characteristics:\n", profitable_customers.describe())
print("\nNon-Profitable Customer Characteristics:\n", non_profitable_customers.describe())

print("\nCorrelation Analysis for Profitable Segments:\n", profitable_corr)
print("\nCorrelation Analysis for Non-Profitable Segments:\n", non_profitable_corr)

#%%
# Evaluate loss ratios for various percentiles
percentiles = [0.05, 0.10, 0.15, 0.20]
thresholds = data['net_contribution'].quantile(percentiles)
loss_ratios = []

for threshold in thresholds:
    high_risk_customers = data[data['net_contribution'] <= threshold]
    loss_ratio = high_risk_customers['total_claimed_amount'].sum() / high_risk_customers['total_premiums_paid'].sum()
    loss_ratios.append(loss_ratio)

print(f"Thresholds: {thresholds}")
print(f"Loss Ratios: {loss_ratios}")
#%%
# Calculate loss ratio for each customer
data['loss_ratio'] = data['total_claimed_amount'] / data['total_premiums_paid']

# Visualize the distribution of loss ratios
plt.figure(figsize=(10, 6))
sns.histplot(data['loss_ratio'], bins=30, kde=True)
plt.title('Distribution of Loss Ratios')
plt.xlabel('Loss Ratio')
plt.ylabel('Frequency')
plt.show()

# Determine the threshold for high loss ratios
high_loss_ratio_threshold = data['loss_ratio'].quantile(0.90)
print(f"High Loss Ratio Threshold (Top 10%): {high_loss_ratio_threshold}")

#%%
# Define a range of thresholds for sensitivity analysis
threshold_range = np.linspace(data['net_contribution'].quantile(0.05), data['net_contribution'].quantile(0.20), 10)

# Analyze the impact of each threshold
for threshold in threshold_range:
    high_risk_customers = data[data['net_contribution'] < threshold]
    num_high_risk_customers = high_risk_customers.shape[0]
    avg_loss_ratio = high_risk_customers['loss_ratio'].mean()
    print(f"Threshold: {threshold:.2f}, Number of High-Risk Customers: {num_high_risk_customers}, Average Loss Ratio: {avg_loss_ratio:.2f}")
#%%
# Select relevant features for clustering
risk_features = ['net_contribution', 'loss_ratio', 'total_claimed_amount', 'total_premiums_paid']

# Standardize the features
scaler = StandardScaler()
risk_features_scaled = scaler.fit_transform(data[risk_features])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['risk_cluster'] = kmeans.fit_predict(risk_features_scaled)

# Analyze the clusters
cluster_summary = data.groupby('risk_cluster')[risk_features].mean()
print("\nCluster Summary:\n", cluster_summary)

# Identify the high-risk cluster
high_risk_cluster = cluster_summary['loss_ratio'].idxmax()
print(f"High Risk Cluster: {high_risk_cluster}")

# Set high-risk threshold based on the cluster
data['high_risk'] = data['risk_cluster'] == high_risk_cluster

#%%
# Assuming `data` is already loaded
# Define relevant features and categorical columns
relevant_features = ['cust_age', 'ins_deductible', 'annual_prem', 'num_vehicles_involved',
                     'bodily_injuries', 'total_claim_amount', 'injury_claim',
                     'property_claim', 'vehicle_claim', 'production_year']
categorical_cols = ['cust_region', 'insured_sex', 'edu_lvl', 'marital_status', 'claim_type',
                    'acc_type', 'incident_city', 'property_damage', 'police_report_avlbl',
                    'car_brand', 'car_model']

# Check which categorical columns are present in the DataFrame
present_categorical_cols = [col for col in categorical_cols if col in data.columns]

# Convert categorical features to numerical using pd.get_dummies() for only present columns
data = pd.get_dummies(data, columns=present_categorical_cols, drop_first=True)

# Extract features for clustering
dummy_cols = data.columns[data.columns.str.startswith(tuple(present_categorical_cols))]
features = data[relevant_features + list(dummy_cols)]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(features_scaled)

# Analyze segment characteristics - only numeric columns
numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
segment_summary = data.groupby('cluster')[numeric_cols].mean()
print("\nSegment Characteristics:\n", segment_summary)

# Visualize the clusters using the first two PCA components
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=data['cluster'], palette='viridis')
plt.title('Customer Segments Based on Profitability')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend(title='Cluster')
plt.show()

#%%
from sklearn.ensemble import RandomForestRegressor
# Define the chosen threshold for high-risk customers

chosen_threshold =  -58289.088980   # 20th percentile
data['high_risk'] = data['net_contribution'] <= chosen_threshold

X = features
y = data['net_contribution']

# Develop a more sophisticated risk assessment model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict risk (net contribution) for each customer
data['predicted_net_contribution'] = model.predict(X)

# Apply more aggressive premium adjustments for high-risk customers
def adjust_premium_high_risk(row):
    base_premium = row['annual_prem']
    risk_factor = row['predicted_net_contribution']
    adjusted_premium = base_premium * (1 + (risk_factor / 60000))  # More aggressive factor for higher risk
    return max(adjusted_premium, base_premium * 1.3)  # Ensure a minimum increase of 30%

# Apply dynamic pricing adjustments with a minimum premium floor
def comprehensive_adjust_premium(row):
    base_premium = row['annual_prem']
    risk_factor = row['predicted_net_contribution']

    if row['high_risk']:  # High-risk customers
        adjusted_premium = adjust_premium_high_risk(row)
    else:  # Non high-risk customers
        if row['risk_cluster'] == 1:  # High-risk cluster
            adjusted_premium = base_premium * (1 + (risk_factor / 70000))
        elif row['risk_cluster'] == 0:  # Moderate-risk cluster
            adjusted_premium = base_premium * (1 + (risk_factor / 80000))
        else:  # Low-risk cluster
            adjusted_premium = base_premium * (1 + (risk_factor / 90000))

    # Ensure premiums do not fall below 95% of the base premium and apply loyalty discount and fraud adjustment
    return max(adjusted_premium, base_premium * 0.95)

data['adjusted_annual_prem'] = data.apply(comprehensive_adjust_premium, axis=1)

# Compare original and adjusted premiums
plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='annual_prem', color='blue', label='Original Premium', kde=True)
sns.histplot(data=data, x='adjusted_annual_prem', color='red', label='Adjusted Premium', kde=True)
plt.title('Comparison of Original and Adjusted Annual Premiums')
plt.xlabel('Annual Premium')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Display summary statistics for original and adjusted premiums
print("\nSummary Statistics for Original and Adjusted Premiums:\n",
      data[['annual_prem', 'adjusted_annual_prem']].describe())

# Calculate Revenues
original_revenue = data['annual_prem'].sum()
adjusted_revenue = data['adjusted_annual_prem'].sum()
print(f"Original Revenue: {original_revenue}")
print(f"Adjusted Revenue: {adjusted_revenue}")
#%%
# Calculate Revenues
original_revenue = data['annual_prem'].sum()
adjusted_revenue = data['adjusted_annual_prem'].sum()
print(f"Original Revenue: {original_revenue}")
print(f"Adjusted Revenue: {adjusted_revenue}")

# Calculate the revenue increase percentage
revenue_increase_percentage = ((adjusted_revenue - original_revenue) / original_revenue) * 100
print(f"Revenue Increase Percentage: {revenue_increase_percentage:.2f}%")
