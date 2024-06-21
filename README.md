# Customer Segmentation, Fraud Detection for Nonlife Insurance Company

## Objective
The primary objective of this project is to conduct an in-depth analysis of claims data from a non-life insurance company focusing on motor insurance. The overarching goal is to leverage advanced data analysis techniques to yield a nuanced understanding of customer profiles, claim patterns, and potential areas of risk. The insights derived are expected to provide actionable insights for risk assessment, customer segmentation, and fraud detection in the insurance domain. 

## Methodology
The methodology involves several distinct stages: data preparation, exploratory data analysis (EDA), feature engineering, segmentation model development, and anomaly detection. Each stage employs specific techniques and focuses on extracting meaningful insights from the data.

1. **Data Collection and Preparation**
2. **Initial Data Exploration**
- Dataset Snapshot: Display the first few records using head() and print the data types of each column with dtypes to understand the dataset structure and verify data formats.
- Unique Value Overview: Use a dictionary comprehension to extract and print unique values from each column, providing a comprehensive overview of the range and categories present in the data.
- Missing Values Analysis: Identify missing values across the dataset using isnull().sum(). 
- Descriptive Statistics: Generate descriptive statistics with describe(include='all') to summarize both numerical and categorical columns, to analyse the central tendencies, dispersion, and potential outliers.
  
3. **Data Cleaning and Imputation**
- Impute Missing Values: Address missing values in the acc_type, emg_services_notified, property_damage, and police_report_avlbl columns using context-specific imputation strategies and mode imputation.
- Standardize Missing Data Representation: Convert placeholders like empty strings and ? to np.nan for consistent handling of missing values.
4. **Feature Engineering**
- Age Group Segmentation: Create a new variable, age_group, by binning the cust_age variable into categorical age ranges.
- Derived Feature Creation: Calculate claim_amount_per_vehicle by dividing total_claim_amount by num_vehicles_involved to gain insights into the average claim cost per vehicle.
- Date Feature Extraction: Transform date features to datetime objects and create a new feature, time_to_claim, representing the time difference between policy initiation and claim reporting.
5. **Data Visualization**
- Distribution of Age Groups: Create a histogram to visualize the distribution of customer ages segmented into predefined age groups.
- Total Claim Amount by Customer Region: Generate a box plot to display the distribution of total claim amounts across different customer regions, highlighting central tendencies, dispersion, and outliers.
6. **Advanced Analysis and Preprocessing**
- Cross Tabulation: Construct contingency tables to examine relationships between categorical variables and key features such as property damage, emergency services notification, and police report availability.
- Standardization and Encoding: Standardize numerical features using StandardScaler and one-hot encode categorical variables using OneHotEncoder to prepare the data for machine learning models.
- Dimensionality Reduction: Apply Principal Component Analysis (PCA) to reduce dimensionality and retain essential variance, to improve computational efficiency and mitigate the curse of dimensionality.
7. **Anomaly Detection and SHAP Analysis**
- Isolation Forest Model: Use an Isolation Forest model for anomaly detection, focusing on identifying outliers in high-dimensional datasets.
- Anomaly Score Assignment: Train the model and predict anomalies, assigning scores to data points for further analysis.
- Top Anomalies Identification: Extract and analyze the top anomalies based on total claim amounts to identify significant outliers.
- SHAP Analysis: Perform SHAP analysis to interpret model predictions and identify features contributing to the likelihood of fraudulent behavior.
8. **Model Building and Interpretation**
- Data Preprocessing for Clustering: Scale numerical variables and apply PCA for dimensionality reduction. One-hot encode categorical variables for clustering.
- K-means Clustering: Apply the K-means algorithm to partition the data into distinct clusters, using the elbow method and silhouette analysis to determine the optimal number of clusters.
- Cluster Visualization and Interpretation: Visualize customer segments, analyze feature distributions across segments, and interpret cluster characteristics to inform risk assessment and fraud detection strategies.



