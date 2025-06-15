# Telco Customer Churn Prediction

## Project Overview

This repository documents an end-to-end machine learning project focused on predicting customer churn for a fictional telecommunications company. The primary objective is to develop a high-performance classification model that can accurately identify customers who are likely to cancel their service. By leveraging customer data, the model provides valuable insights that can inform proactive retention strategies, ultimately reducing revenue loss and improving customer loyalty.

The project encompasses a full workflow, including data cleaning, in-depth exploratory data analysis, feature engineering, model training, advanced hyperparameter optimization, and final model evaluation and interpretation.

## Dataset Description

The analysis is based on the "Telco Customer Churn" dataset, which contains 7,043 customer records and 21 attributes. These attributes describe the customer's demographic profile, account information, and the services they have subscribed to.

### Data Dictionary

| **Category**                     | **Variable Name**  | **Description**                                                                                     |
| -------------------------------- | ------------------ | --------------------------------------------------------------------------------------------------- |
| **Demographic Information**      | `gender`           | The customer's gender (Male / Female).                                                              |
|                                  | `SeniorCitizen`    | Indicates if the customer is a senior citizen (1 for Yes, 0 for No).                                |
|                                  | `Partner`          | Indicates if the customer has a partner (Yes / No).                                                 |
|                                  | `Dependents`       | Indicates if the customer has dependents (Yes / No).                                                |
| **Customer Account Information** | `customerID`       | A unique identifier for each customer.                                                              |
|                                  | `tenure`           | The number of months the customer has been with the company.                                        |
|                                  | `Contract`         | The customer's contract term (Month-to-month, One year, Two year).                                  |
|                                  | `PaperlessBilling` | Indicates if the customer has opted for paperless billing (Yes / No).                               |
|                                  | `PaymentMethod`    | The customer's payment method (e.g., Electronic check, Mailed check).                               |
|                                  | `MonthlyCharges`   | The amount charged to the customer monthly.                                                         |
|                                  | `TotalCharges`     | The total amount charged to the customer over their entire tenure.                                  |
| **Subscribed Services**          | `PhoneService`     | Indicates if the customer has phone service (Yes / No).                                             |
|                                  | `MultipleLines`    | Indicates if the customer has multiple phone lines (Yes / No / No phone service).                   |
|                                  | `InternetService`  | The customer's internet service provider (DSL, Fiber optic, No).                                    |
|                                  | `OnlineSecurity`   | Indicates if the customer subscribes to online security (Yes / No / No internet service).           |
|                                  | `OnlineBackup`     | Indicates if the customer subscribes to online backup (Yes / No / No internet service).             |
|                                  | `DeviceProtection` | Indicates if the customer subscribes to device protection (Yes / No / No internet service).         |
|                                  | `TechSupport`      | Indicates if the customer subscribes to tech support (Yes / No / No internet service).              |
|                                  | `StreamingTV`      | Indicates if the customer subscribes to a TV streaming service (Yes / No / No internet service).    |
|                                  | `StreamingMovies`  | Indicates if the customer subscribes to a movie streaming service (Yes / No / No internet service). |
| **Target Variable**              | `Churn`            | **Indicates whether the customer churned (Yes / No). This is the value to be predicted.**           |

## Project Methodology

The project followed a structured, multi-stage methodology to ensure robust and reliable results.

### 1. Exploratory Data Analysis (EDA)

A comprehensive analysis was conducted to understand the data and uncover initial insights. This involved visualizing distributions and relationships between variables.

- **Key Findings:** The analysis revealed that churn is strongly correlated with specific customer attributes. Customers with **month-to-month contracts**, **fiber optic internet service**, and **low tenure** showed a significantly higher propensity to churn. Furthermore, payment via **electronic check** was also associated with a higher churn rate.

### 2. Feature Engineering and Preprocessing

The data was prepared for modeling through a series of transformations:

- **Data Cleaning:** The `TotalCharges` column was converted to a numeric type, and the few resulting missing values were imputed using the median.

- **Feature Creation:** A new feature, `num_support_services`, was engineered to count the number of supplementary protection and support services each customer had, aiming to capture a measure of customer engagement.

- **Preprocessing Pipeline:** A `scikit-learn` `Pipeline` was constructed to streamline transformations. This pipeline uses `StandardScaler` to scale numerical features and `OneHotEncoder` to convert categorical features into a numerical format. This approach ensures consistency and prevents data leakage.

### 3. Modeling and Validation

- **Algorithm Selection:** **XGBoost (Extreme Gradient Boosting)** was chosen as the primary modeling algorithm due to its proven high performance on structured, tabular data.

- **Baseline Model:** A baseline XGBoost model with default parameters was trained and evaluated using **5-fold Stratified Cross-Validation**. This provided a reliable initial performance benchmark, achieving a mean ROC AUC score of approximately **0.82**.

### 4. Hyperparameter Optimization

To maximize model performance, the **Optuna** library was employed to conduct an efficient and intelligent search for the optimal XGBoost hyperparameters.

- **Process:** Optuna ran 50 trials, testing different combinations of parameters like `n_estimators`, `learning_rate`, and `max_depth`.

- **Outcome:** This optimization process successfully improved the model's performance, increasing the cross-validated mean ROC AUC score to approximately **0.85**.

## Results and Business Insights

The final optimized model was trained on the full training dataset and evaluated on a held-out test set to simulate real-world performance.

### Performance

- **ROC AUC Score (Test Set):** The final model achieved a score of over **0.86**, indicating excellent capability in distinguishing between churning and non-churning customers.

- **Classification Metrics:** The model demonstrated a strong balance between **precision** (minimizing false positives) and **recall** (identifying a high percentage of actual churners), making it a valuable tool for a targeted retention campaign.

### Key Churn Drivers (Feature Importance)

The model identified the following factors as the most influential predictors of churn:

1. **Contract Type (Month-to-month)**

2. **Customer Tenure**

3. **Internet Service (Fiber Optic)**

4. **Monthly Charges**

### Actionable Business Recommendations

Based on the model's findings, the following strategic actions are recommended:

- **Enhance Long-Term Contracts:** Develop marketing campaigns and incentives to encourage customers on month-to-month plans to switch to one- or two-year contracts.

- **Focus on New Customer Retention:** Implement a dedicated onboarding program and regular check-ins for new customers (low tenure), as they are at the highest risk.

- **Investigate Fiber Optic Service:** Analyze the reasons behind the high churn rate for fiber optic customers. This could be related to pricing, service stability, or perceived value. Addressing these issues could significantly reduce churn.

## Technologies Used

- **Language:** Python 3

- **Libraries:**
  
  - **Pandas & NumPy:** Data manipulation and numerical operations.
  
  - **Matplotlib & Seaborn:** Data visualization.
  
  - **Scikit-Learn:** Data preprocessing, pipelines, and model evaluation.
  
  - **XGBoost:** Core machine learning algorithm.
  
  - **Optuna:** Hyperparameter optimization.

- **Environment:** Jupyter Notebook

## How to Run the Project

1. **Clone the Repository:**
   
   ```
   git clone [YOUR_REPOSITORY_URL]
   ```

2. **Navigate to the Project Directory:**
   
   ```
   cd [repository-name]
   ```

3. **(Recommended) Create and Activate a Virtual Environment:**
   
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. **Install Dependencies:**
   
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna jupyter
   ```

5. **Launch Jupyter Notebook:**
   
   ```
   jupyter notebook
   ```

6. Open the `.ipynb` file and execute the cells sequentially.