# CreditWise Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved based on applicant financial, demographic, employment, and loan-related information.

The project compares multiple machine learning approaches and evaluates them using accuracy, precision, recall, F1-score, and confusion matrix.

## Project Overview

Loan approval depends on several factors such as income, credit score, existing loans, savings, loan amount, employment status, and other applicant information.

The goal of this project is to predict:

- `0` → Loan Not Approved
- `1` → Loan Approved

## Dataset Features

The dataset contains the following features:

| Feature | Type | Description |
|---|---|---|
| `Applicant_ID` | Identifier | Unique applicant identifier; removed before modeling |
| `Applicant_Income` | Numerical | Applicant's income |
| `Coapplicant_Income` | Numerical | Coapplicant's income |
| `Employment_Status` | Categorical | Applicant's employment type |
| `Age` | Numerical | Applicant's age |
| `Marital_Status` | Binary Categorical | Single / Married |
| `Dependents` | Numerical | Number of dependents |
| `Credit_Score` | Numerical | Applicant's credit score |
| `Existing_Loans` | Numerical | Number of existing loans |
| `DTI_Ratio` | Numerical | Debt-to-income ratio |
| `Savings` | Numerical | Applicant's savings |
| `Collateral_Value` | Numerical | Value of collateral |
| `Loan_Amount` | Numerical | Requested loan amount |
| `Loan_Term` | Numerical | Loan repayment term |
| `Loan_Purpose` | Categorical | Purpose of the loan |
| `Property_Area` | Categorical | Rural / Semiurban / Urban |
| `Education_Level` | Binary Categorical | Graduate / Not Graduate |
| `Gender` | Binary Categorical | Male / Female |
| `Employer_Category` | Categorical | Employer category |
| `Loan_Approved` | Binary Target | Yes / No |

## Data Preprocessing

### 1. Missing Value Handling

Missing values were handled separately according to the feature type.

**Categorical features:**

Missing categorical values were filled using the **mode**, which is the most frequently occurring value.

```python
for col in categorical_cols:
    loan_data[col] = loan_data[col].fillna(loan_data[col].mode()[0])
```

**Numerical features:**

Missing numerical values were filled using the **median**.

```python
for col in numerical_cols:
    loan_data[col] = loan_data[col].fillna(loan_data[col].median())
```

The median was used for numerical features to provide a robust value when the data may contain outliers or skewness.

### 2. Removing Applicant ID

`Applicant_ID` was removed because it is an identifier and does not represent a meaningful predictive feature.

```python
loan_data.drop("Applicant_ID", axis=1, inplace=True)
```

### 3. Converting Binary Categorical Features

Binary categorical features were converted into numerical values using `map()`.

For example, `Gender` was converted as:

```python
loan_data["Gender"] = loan_data["Gender"].map({
    "Female": 0,
    "Male": 1
})
```

Similarly:

```python
loan_data["Marital_Status"] = loan_data["Marital_Status"].map({
    "Single": 0,
    "Married": 1
})
```

Education level was converted as:

```python
loan_data["Education_Level"] = loan_data["Education_Level"].map({
    "Not Graduate": 0,
    "Graduate": 1
})
```

The target variable was also converted into binary form:

```python
loan_data["Loan_Approved"] = loan_data["Loan_Approved"].map({
    "No": 0,
    "Yes": 1
})
```

This converts categorical values into numerical representations that machine learning algorithms can process.

### 4. One-Hot Encoding

Nominal categorical features with more than two categories were converted using one-hot encoding.

The following features were one-hot encoded:

- `Employment_Status`
- `Loan_Purpose`
- `Property_Area`
- `Employer_Category`

```python
loan_data = pd.get_dummies(
    loan_data,
    columns=nominal_cols,
    drop_first=True
)
```

`drop_first=True` removes one category from each encoded feature to avoid redundant dummy variables.

### 5. Train-Test Split

The dataset was divided into training and testing sets using an 80:20 split.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

- 80% → Training data
- 20% → Testing data

### 6. Feature Scaling

Numerical features were standardized using `StandardScaler`.

```python
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

Standardization transforms the features so that they are approximately centered around mean `0` with standard deviation `1`.

The scaler is **fitted only on the training data** and then used to transform the test data.

## Machine Learning Models

The project evaluates:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Logistic Regression with L2 regularization
- Logistic Regression with L1 regularization
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)

Cross-validation and hyperparameter tuning were also explored for regularized models and KNN.

## Hyperparameter Tuning

### KNN

Different values of `n_neighbors` were evaluated using 5-fold cross-validation.

```python
param_grid = {
    "n_neighbors": [3, 5, 7, 9]
}
```

Recall was used as the scoring metric during the KNN hyperparameter search.

### Lasso

Different `alpha` values were evaluated using `GridSearchCV`:

```python
param_grid = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}
```

The best `alpha` was selected using 5-fold cross-validation.

### Ridge

Ridge used the same candidate `alpha` values:

```python
param_grid = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}
```

The best `alpha` was selected using 5-fold cross-validation.

## Model Evaluation

Classification models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

For the regression-based experiments, the continuous predictions were converted into binary classes using a threshold of `0.5` so that classification metrics could be compared experimentally.

```python
y_pred_class = (y_predict >= 0.5).astype(int)
```

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Linear Regression | 85.00% | 75.41% | 75.41% | 75.41% |
| Lasso Regression | 85.50% | 78.57% | 72.13% | 75.21% |
| Ridge Regression | 85.00% | 75.41% | 75.41% | 75.41% |
| Logistic Regression (L2) | 86.50% | 78.33% | 77.05% | 77.69% |
| **Logistic Regression (L1)** | **87.00%** | **78.69%** | **78.69%** | **78.69%** |
| Gaussian Naive Bayes | 86.50% | 80.36% | 73.77% | 76.92% |
| KNN | 74.50% | 58.06% | 59.02% | 58.54% |

## Best Model

**Logistic Regression with L1 regularization** achieved the best overall classification performance among the evaluated models.

### Performance

- **Accuracy:** 87.00%
- **Precision:** 78.69%
- **Recall:** 78.69%
- **F1 Score:** 78.69%

The model was implemented using:

```python
LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=1000000
)
```

L1 regularization was used to control model complexity and can encourage some feature coefficients toward zero.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Missing Value Handling
   ↓
Remove Identifier
   ↓
Binary Categorical Mapping
   ↓
One-Hot Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Cross-Validation / Hyperparameter Tuning
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model: Logistic Regression (L1)
```

## Project Structure

```text
CreditWise-Loan-Approval-Prediction/
│
├── Credit_Wise_Loan_Approval.ipynb
├── loan_approval_data.csv
└── README.md
```

## Key Learning Outcomes

Through this project, I worked with:

- Data preprocessing
- Missing-value handling
- Numerical and categorical feature identification
- Binary categorical encoding using mapping
- One-hot encoding
- Feature scaling using StandardScaler
- Train-test splitting
- Logistic Regression
- L1 and L2 regularization
- Linear Regression
- Ridge and Lasso Regression
- Gaussian Naive Bayes
- KNN classification
- Cross-validation
- GridSearchCV
- Hyperparameter tuning
- Classification metrics
- Confusion matrix
- Model comparison

## Conclusion

Among the evaluated models, **Logistic Regression with L1 regularization** provided the strongest overall classification performance, achieving **87% accuracy, 78.69% precision, 78.69% recall, and 78.69% F1-score** on the test set.

The project demonstrates an end-to-end machine learning workflow covering data preprocessing, categorical encoding, feature scaling, model training, hyperparameter tuning, evaluation, and model comparison.