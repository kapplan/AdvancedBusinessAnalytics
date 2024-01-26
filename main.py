import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import chi2

# Load the dataset
file_path = '/Users/apple/Downloads/claims_q12023.csv'  # Update this to your file's path
data = pd.read_csv(file_path, delimiter=';', header=0)

print(data.head())

column_data_types = data.dtypes
print(column_data_types)

#Convert Date Columns
data['coverage_start_date'] = pd.to_datetime(data['coverage_start_date'], format='%d.%m.%Y')
data['claim_incurred_date'] = pd.to_datetime(data['claim_incurred_date'], format='%d.%m.%Y')

#Convert categorical columns
categorical_columns = ['cust_region', 'sum_assured_group', 'insured_sex', 'edu_lvl',
                       'marital_status', 'claim_type', 'acc_type', 'emg_services_notified',
                       'incident_city', 'property_damage', 'police_report_avlbl',
                       'car_brand', 'car_model']

for col in categorical_columns:
    data[col] = data[col].astype('category')

print(data.dtypes)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Generate descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# Visualizing the missingness pattern
msno.matrix(data)
plt.show()

sorted=data.sort_values('emg_services_notified')
msno.matrix(sorted)
plt.show()

# Heatmap to show correlations of missingness between different columns
msno.heatmap(data)
plt.show()

# Bar chart to show the count of missing values in each column
msno.bar(data)
plt.show()

# Create dummy variables for categorical data
categorical_columns = data.select_dtypes(['category']).columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)


data_encoded['coverage_start_year'] = data['coverage_start_date'].dt.year
data_encoded['coverage_start_month'] = data['coverage_start_date'].dt.month
data_encoded['claim_incurred_year'] = data['claim_incurred_date'].dt.year
data_encoded['claim_incurred_month'] = data['claim_incurred_date'].dt.month

# Optionally, calculate durations or intervals if relevant
current_date = pd.Timestamp('now')
data_encoded['days_since_coverage_start'] = (current_date - data['coverage_start_date']).dt.days
data_encoded['days_since_claim_incurred'] = (current_date - data['claim_incurred_date']).dt.days
data = data.drop('coverage_start_date', axis=1)
print(data_encoded.dtypes)
print(data_encoded)

columns_with_missing_values = data.columns[data.isnull().any()].tolist()
print("Columns with missing values:", columns_with_missing_values)

data_with_missing = data[columns_with_missing_values]

# Assuming 'data' is your DataFrame and 'acc_type' is the categorical column
data_encoded = pd.get_dummies(data_with_missing, drop_first=True)

#Little's MCAR Test
def little_mcar_test(data, alpha=0.05):
    """
    Performs Little's MCAR (Missing Completely At Random) test on a dataset with missing values.

    Parameters:
    data (DataFrame): A pandas DataFrame with n observations and p variables, where some values are missing.
    alpha (float): The significance level for the hypothesis test (default is 0.05).

    Returns:
    A tuple containing:
    - A matrix of missing values that represents the pattern of missingness in the dataset.
    - A p-value representing the significance of the MCAR test.
    """

    # Calculate the proportion of missing values in each variable
    p_m = data.isnull().mean()

    # Calculate the proportion of complete cases for each variable
    p_c = data.dropna().shape[0] / data.shape[0]

    # Calculate the correlation matrix for all pairs of variables that have complete cases
    R_c = data.dropna().corr()

    # Calculate the correlation matrix for all pairs of variables using all observations
    R_all = data.corr()

    # Calculate the difference between the two correlation matrices
    R_diff = R_all - R_c

    # Calculate the variance of the R_diff matrix
    V_Rdiff = np.var(R_diff, ddof=1)

    # Calculate the expected value of V_Rdiff under the null hypothesis that the missing data is MCAR
    E_Rdiff = (1 - p_c) / (1 - p_m).sum()

    # Calculate the test statistic
    T = np.trace(R_diff) / np.sqrt(V_Rdiff * E_Rdiff)

    # Calculate the degrees of freedom
    df = data.shape[1] * (data.shape[1] - 1) / 2

    # Calculate the p-value using a chi-squared distribution with df degrees of freedom and the test statistic T
    p_value = 1 - chi2.cdf(T ** 2, df)

    # Create a matrix of missing values that represents the pattern of missingness in the dataset
    missingness_matrix = data.isnull().astype(int)

    # Return the missingness matrix and the p-value
    return missingness_matrix, p_value

little_mcar_test_result = little_mcar_test(data_encoded)
print("Missingness Matrix:\n", little_mcar_test_result[0])
print("MCAR Test p-value:", little_mcar_test_result[1])


# A low p-value (typically <0.05) would suggest
# that the data is not missing completely at random.

# Create a binary indicator (0 or 1) for missingness
for col in data.columns:
    data[col + '_missing'] = data[col].isnull().astype(int)

# Perform Chi-squared tests
for col in data.columns:
    if data[col].dtype == 'object':  # For categorical columns
        for missing_col in [col2 for col2 in data.columns if '_missing' in col2]:
            contingency_table = pd.crosstab(data[col], data[missing_col])
            chi2, p, dof, ex = chi2_contingency(contingency_table, correction=False)
            print(f"Chi-squared test for {col} and {missing_col}: p-value = {p}")