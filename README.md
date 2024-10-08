# Customer Segmentation and Fraud Detection for Nonlife Insurance Company

## Objective
The primary objective of this project is to conduct an in-depth analysis of claims data from a non-life insurance company focusing on motor insurance. The overarching goal is to leverage advanced data analysis techniques to yield a nuanced understanding of customer profiles, claim patterns, and potential areas of risk in order to optimize profit. 

## Dataset
The dataset contains 1000 rows and 29 columns. Following are the columns:
- cust_age: int64 - Age of the policyholder.
- policy_id: int64 - Unique identifier for each insurance policy.
- coverage_start_date: object - Date when the insurance coverage began.
- cust_region: object - Geographic region of the policyholder.
- sum_assured_group: object - Sum assured groups indicating coverage amount.
- ins_deductible: int64 - Deductible amount on the insurance policy.
- annual_prem: float64 - Annual premium paid by the policyholder.
- zip_code: int64 - ZIP code of the policyholder's residence.
- insured_sex: object - Gender of the insured.
- edu_lvl: object - Education level of the policyholder.
- marital_status: object - Marital status of the policyholder.
- claim_incurred_date: object - Date when the claim was incurred.
- claim_type: object - Types of claims (e.g., theft, accident).
- acc_type: object - Nature of the accident.
- emg_services_notified: object - Whether emergency services were notified.
- incident_city: object - City where the incident occurred.
- incident_hour: int64 - Hour of the day when the incident occurred.
- num_vehicles_involved: int64 - Number of vehicles involved in the incident.
- property_damage: object - Whether property damage occurred.
- bodily_injuries: int64 - Number of bodily injuries reported.
- witnesses: int64 - Number of witnesses to the incident.
- police_report_avlbl: object - Whether a police report is available.
- total_claim_amount: int64 - Total amount claimed.
- injury_claim: int64 - Amount claimed for injuries.
- property_claim: int64 - Amount claimed for property damage.
- vehicle_claim: int64 - Amount claimed for vehicle damage.
- car_brand: object - Brand of the car involved in the claim.
- car_model: object - Model of the car involved.
- production_year: int64 - Production year of the car.

 ## Methodology
The methodology involves several distinct stages: data preparation, exploratory data analysis (EDA), feature engineering, segmentation model development, anomaly detection, and revenue optimization through dynamic pricing. Each stage employs specific techniques and focuses on extracting meaningful insights from the data.

1. **Data Collection and Preparation**
3. **Initial Data Exploration**
- Dataset Snapshot: Understand the dataset structure and verify data formats.
- Unique Value Overview: Use a dictionary comprehension to extract and print unique values from each column, providing a comprehensive overview of the range and categories present in the data.
- Missing Values Analysis: Identify missing values across the dataset. 
- Descriptive Statistics: Generate descriptive statistics with describe(include='all') to summarize both numerical and categorical columns, to analyse the central tendencies, dispersion, and potential outliers.
  
4. **Data Cleaning and Imputation**
- Missing Data Representation: Convert placeholders like empty strings and ? to np.nan for consistent handling of missing values.
- Missing Values Imputaion: Address missing values in the acc_type, emg_services_notified, property_damage, and police_report_avlbl columns using context-specific imputation strategies and mode imputation.

5. **Feature Engineering**
- Age Group Segmentation: Create a new variable, age_group, by binning the cust_age variable into categorical age ranges.
- New Feature Creation: Calculate claim_amount_per_vehicle by dividing total_claim_amount by num_vehicles_involved to gain insights into the average claim cost per vehicle. Add a new feature profitability 
- Date Feature Extraction: Transform date features to datetime objects and create a new feature, time_to_claim, representing the time difference between policy initiation and claim reporting.
  
6. **Data Visualization**

7. **Advanced Analysis and Preprocessing**
- **Cross Tabulation**: Construct contingency tables to examine relationships between categorical variables and key features such as property damage, emergency services notification, and police report availability.
- **Standardization and Encoding**: Standardize numerical features using StandardScaler and one-hot encode categorical variables using OneHotEncoder to prepare the data for machine learning models.
- **Dimensionality Reduction**: Apply Principal Component Analysis (PCA) to reduce dimensionality and retain essential variance, to improve computational efficiency and mitigate the curse of dimensionality.
8. **Customer Segmentation**:
- First Approach (**K-Means with The Silhouette Score and The Elbow Method**) 
- Second Approach (**TruncatedSVD and K-means**)


9. **Anomaly Detection and SHAP Analysis**
- **Isolation Forest Model**: Use an Isolation Forest model for anomaly detection, focusing on identifying outliers.
- Top Anomalies Identification: Extract and analyze the top anomalies to identify significant outliers.
- **SHAP Analysis**: Perform SHAP analysis to interpret model predictions and identify features contributing to the likelihood of fraudulent behavior.
10. **Model Building and Interpretation**
- Data Preprocessing for Clustering: Scale numerical variables and apply PCA for dimensionality reduction. One-hot encode categorical variables for clustering.
- **K-means Clustering**: Apply the K-means algorithm to partition the data into distinct clusters, using the elbow method and silhouette analysis to determine the optimal number of clusters.
- Cluster Visualization and Interpretation: Visualize customer segments, analyze feature distributions across segments, and interpret cluster characteristics to inform prrofitability and high risk assessment.
  
11. **Further Analysis of Profitable and Non-Profitable Segments**
- Summary Statistics: Summary statistics are generated for profitable and non-profitable customers to compare their characteristics.
- Distribution Analysis: The distribution of age groups, claim types, and annual premiums in profitable vs non-profitable segments are visualized using count plots and box plots.
- Correlation Analysis: Correlation matrices are created for profitable and non-profitable segments to identify key relationships between features.
- Evaluate Loss Ratios: Loss ratios are evaluated for various percentiles to determine the threshold for high-risk customers.
- Cluster Analysis: K-Means clustering is applied to identify high-risk clusters based on features such as net contribution, loss ratio, total claimed amount, and total premiums paid.

12. **Revenue Analysis and Dynamic Pricing**
- **Dynamic Pricing Strategy**: A sophisticated risk assessment model using RandomForestRegressor is developed to predict the net contribution of each customer. More aggressive premium adjustments are applied for high-risk customers to optimize revenue.
- **Adjust Premiums for High-Risk Customers**: Premium adjustments are applied dynamically based on the risk assessment model to ensure a minimum increase for high-risk customers and prevent revenue loss.

## Summary and Findings

- Majority of policyholders fall within the '26-35' and '36-45' age groups.
- The median total claim amount is relatively consistent across regions around 58,000 to 60,000, with the 'north' region displaying a slightly higher median compared to 'east' and 'west'.
- 56-65 years age group tend to have higher average claims, while 18-25 comes second. 
- The risk of claims does not increase dramatically at specific times.
- Long-duration policies lead to higher insurance claims.
- The median annual premium paid is approximately 1300. There are some annual premium values significantly lower and higher than the typical range, below 600 and above 1800.
- Most of the net contributions are negative, indicating that the **insurance company pays out more in claims than it collects in premiums for most policies**.
The biggest value is around **-50.000, suggesting that this is the most common loss amount**. Most policies result in losses, **a few results in very high losses**. 
- Higher total premiums paid tend to result in higher net contributions. The positive correlation suggests that **increasing premium amounts could potentially reduce net losses**.


### Policy IDs with significant financial losses:

| Policy ID | Net Contribution | Car Brand  | Car Model  |
|-----------|------------------|------------|------------|
| 132045    | -151509.365038   | Subaru     | Impreza    |
| 209446    | -140091.436277   | Mercedes   | E400       |
| 310312    | -138950.512690   | BMW        | X5         |
| 113947    | -137967.702177   | Chevrolet  | Tahoe      |
| 453588    | -135790.960383   | Mercedes   | ML350      |
| 113442    | -105828.234086   | Volkswagen | Passat     |
| 226009    | -98255.107844    | Dodge      | RAM        |
| 301512    | -97926.774949    | Audi       | A5         |
| 130053    | -96168.230226    | Audi       | A3         |
| 484930    | -94934.247392    | Mercedes   | ML350      |
| 264572    | -93931.659849    | Honda      | CRV        |
| 574854    | -92630.272033    | Ford       | Escape     |
| 371084    | -91633.887967    | BMW        | X5         |
| 565225    | -90372.978508    | Chevrolet  | Malibu     |
| 359353    | -89720.998522    | Accura     | TL         |
| 214333    | -88073.603258    | Toyota     | Corolla    |
| 555873    | -86640.002765    | Audi       | A5         |
| 215236    | -84435.630883    | Audi       | A5         |
| 101040    | -83388.479671    | Subaru     | Legacy     |
| 147217    | -83181.582642    | Jeep       | Wrangler   |



### Profitable/Non-Profitable High/Low Risk Customer Segmentation
![Diagram](custsomer_segmentation_outliers.jpg)

| Risk Cluster | Net Contribution | Total Premiums Paid |
|--------------|------------------:|--------------------:|
| 0            | -36026.40         | 27390.56            |
| 1            | -55018.73         | 9851.43             |
| 2            | 9412.76           | 17868.08            |

High Risk Cluster: 1

### After applying more aggressive premium adjustments to high-risk customers:

![Diagram](adjusted_pricing.jpg)

**Original Revenue**: 1254516.15

**Adjusted Revenue After Dynamic Pricing based on Risk**: 1320695.395160412

**Revenue Increase in Percentage**: 5.28%

### Summary Statistics for Original and Adjusted Premiums

| Statistic | Annual Premium | Adjusted Annual Premium |
|-----------|----------------:|------------------------:|
| Count     | 1000.000000     | 1000.000000             |
| Mean      | 1254.516150     | 1320.695395             |
| Std Dev   | 244.167395      | 315.827434              |
| Min       | 431.440000      | 449.923846              |
| 25%       | 1087.717500     | 1109.148750             |
| 50%       | 1255.310000     | 1285.462112             |
| 75%       | 1413.805000     | 1497.675066             |
| Max       | 2045.700000     | 2659.410000             |

# Fraud Analysis

**Susipicous Claims Policy IDs**: 113947, 310312, 592543, 132045, 438097, 500283, 185750, 209446, 113442, 453588

**Total Net Contribution (Claim Amount substracted by Annual Premiums Paid) of Suspicious Policy IDs**: -1192486.0300000003


