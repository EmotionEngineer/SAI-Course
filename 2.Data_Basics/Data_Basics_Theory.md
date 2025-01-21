# 🗃️ Data Basics

Data is the foundation of all AI systems. In this lesson, you’ll explore the role of data in AI, types of data, and essential data preprocessing techniques for effective AI model training.

---

## 📌 What is Data?

In AI, **data** represents any information that can be collected, measured, and analyzed. It’s the primary input for machine learning algorithms, which use data to learn patterns and make predictions.

### 🔍 Key Properties of Data:
- **Volume**: Quantity of data, from small datasets to large-scale data.
- **Variety**: Different types and formats of data (e.g., text, images, audio).
- **Velocity**: The speed at which data is generated or collected.
- **Veracity**: The reliability and accuracy of the data.

---

## 🌐 Types of Data in AI

### 1. **Structured Data**
   - Data organized in a predefined format, typically in rows and columns.
   - Examples: **Spreadsheets**, **SQL Databases**.
   - **Example Schema**:
     ```text
     | ID  | Name   | Age | City      |
     | --- | ------ | --- | --------- |
     | 1   | Alice  | 25  | New York  |
     | 2   | Bob    | 30  | Chicago   |
     ```

### 2. **Unstructured Data**
   - Data without a specific format, often text-heavy or multimedia.
   - Examples: **Emails**, **Social Media Posts**, **Images**.

### 3. **Semi-Structured Data**
   - Data that doesn’t fit neatly into structured formats but has some organization.
   - Examples: **XML files**, **JSON files**.

### 4. **Time-Series Data**
   - Data points collected at specific time intervals.
   - Examples: **Stock Prices**, **Weather Data**.

---

## 🧹 Data Processing

Before data can be used in AI, it needs to be **processed** to make it clean and usable for modeling.

### Key Steps in Data Processing:
1. **Data Collection**: Gathering relevant data from various sources.
2. **Data Cleaning**: Removing errors, duplicates, and inconsistencies.
3. **Data Transformation**: Converting formats or encoding variables.
4. **Data Splitting**: Dividing data into training, validation, and test sets.

> **Fun Fact**: Data scientists spend about 80% of their time preparing and cleaning data!

---

## 🔧 Essential Tools for Data Handling

### Popular Libraries for Data Processing in Python:
- **Pandas**: Ideal for tabular data manipulation.
- **NumPy**: Provides support for large, multi-dimensional arrays and matrices.
- **Matplotlib & Seaborn**: Libraries for visualizing data.

---

## 📊 Data Preprocessing Techniques

### **1. Normalization and Scaling**

Normalization and scaling are essential preprocessing steps that adjust data to a common scale without distorting differences in the range of values. These techniques are especially useful for algorithms that rely on distance measurements, such as K-Nearest Neighbors (KNN) and Support Vector Machines (SVM).

---

#### **Normalization (Min-Max Scaling)**

Normalization, or **Min-Max Scaling**, transforms the data to fit within a specific range, typically between 0 and 1. This process adjusts each feature to a common scale while preserving relationships in the original data.

The **formula** for normalization is:

$$
x_{\text{normalized}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}
$$

where:
- $x$ is the original value,
- $x_{\text{min}}$ and $x_{\text{max}}$ are the minimum and maximum values in the data.

This transformation scales all values between 0 and 1, where the smallest value becomes 0 and the largest value becomes 1.

**Example in Python:**

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[['Column_Name']] = scaler.fit_transform(data[['Column_Name']])
```

---

#### **Standardization (Z-score Scaling)**

Standardization, also known as **Z-score Scaling**, adjusts data based on the mean and standard deviation, transforming it to have a mean of 0 and a standard deviation of 1. This technique is useful for data with a Gaussian (normal) distribution.

The **formula** for standardization is:

$$
z = \frac{x - \mu}{\sigma}
$$

where:
- $x$ is the original value,
- $\mu$ is the mean of the data,
- $\sigma$ is the standard deviation of the data.

After standardization, most values will lie between -3 and 3, following the properties of a normal distribution.

**Example in Python:**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['Column_Name']] = scaler.fit_transform(data[['Column_Name']])
```

---

### **When to Use Normalization vs. Standardization**

- **Normalization**: Ideal when the data does not follow a Gaussian distribution and has varied ranges, especially for algorithms that require data within a bounded range.
- **Standardization**: Best suited for data that approximates a normal distribution, making it beneficial for many machine learning algorithms.

Choosing between normalization and standardization depends on the distribution and the algorithm's sensitivity to feature scales.

### 2. Encoding Categorical Variables

Converting non-numeric data, such as categories, into numeric form for model compatibility.

> **Example Code**:

```python
import pandas as pd

# Sample data
data = pd.DataFrame({'Color': ['Red', 'Green', 'Blue', 'Green']})

# One-hot encoding
data_encoded = pd.get_dummies(data, columns=['Color'])
```

---

### **3. Handling Missing Values**

Missing data is a common issue in datasets and can impact the accuracy of our models. Common imputation techniques include using the **mean**, **median**, or a **constant value** to fill in missing values.

---

#### **Mean Imputation**

For **mean imputation**, we replace missing values with the **average** of the non-missing values in the column. This method is effective when the data is symmetrically distributed without significant outliers.

The **formula** for mean imputation is:

$$
\text{Imputed Value} = \frac{\sum_{i=1}^{n} x_i}{n}
$$

where:
- $x_i$ are the observed values (excluding missing values),
- $n$ is the count of non-missing values.

**Example in Python:**

```python
# Impute missing values with the mean of the column
data['Column_Name'].fillna(data['Column_Name'].mean(), inplace=True)
```

---

#### **Median Imputation**

**Median imputation** is often preferred for skewed data or when outliers are present, as the median is less affected by extreme values.

The **formula** for median imputation is:

$$
\text{Imputed Value} = 
\begin{cases} 
      x_{\left(\frac{n+1}{2}\right)} & \text{if } n \text{ is odd} \\
      \frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2} + 1\right)}}{2} & \text{if } n \text{ is even} 
\end{cases}
$$

where:
- $n$ is the count of non-missing values,
- $x_{(i)}$ represents the $i$-th value when sorted.

**Example in Python:**

```python
# Impute missing values with the median of the column
data['Column_Name'].fillna(data['Column_Name'].median(), inplace=True)
```

---

#### **Constant Imputation**

In **constant imputation**, we replace missing values with a specified constant. This method is useful when a particular value, like 0 or -1, has specific meaning in the context of the data, or if you want a distinct value to signal imputation. 

For example, if using **0** as a constant imputation value:

**Example in Python:**

```python
# Impute missing values with a constant value, e.g., 0
data['Column_Name'].fillna(0, inplace=True)
```

---

### **When to Use Each Method**

- **Mean Imputation**: Suitable for normally distributed data without significant outliers.
- **Median Imputation**: Best for skewed data or when outliers are present.
- **Constant Imputation**: Ideal when you want a specific value that is contextually meaningful, or when you want to indicate that an imputation occurred.

Selecting the right imputation method depends on the data distribution and the specific requirements of your analysis or model.

### 4. Feature Engineering

Creating new features that may better capture patterns in the data.

> **Example Code**:

```python
import pandas as pd

# Sample data
data = pd.DataFrame({'Height': [1.7, 1.8, 1.6], 'Weight': [65, 80, 70]})

# Feature engineering: calculating BMI
data['BMI'] = data['Weight'] / (data['Height'] ** 2)
```

---

## 🔄 Data Splitting for Model Training

Splitting data into subsets is critical for evaluating AI models. The main splits include:

- **Training Set**: Used to train the AI model.
- **Validation Set**: Used for tuning model parameters.
- **Testing Set**: Final data for evaluating model performance.

> **Example Code**:

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample data
data = pd.DataFrame({'Feature': [1, 2, 3, 4, 5], 'Label': [10, 20, 30, 40, 50]})

# Split data into 80% training and 20% testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
```

---

## 🚀 Why is Data Important in AI?

Data allows AI to **learn patterns**, **generalize to new data**, **provide insights**, and **make decisions**. High-quality data leads to more reliable and accurate AI models.

> **Quote to Remember**:
> *"Data is the new oil."* – Clive Humby

---

## 🎨 Extended Example: Data Processing with Pandas

Here’s a more comprehensive example that incorporates various preprocessing techniques:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Load data
data = pd.DataFrame({
    'Age': [25, None, 30, 35, 45],
    'Height': [1.7, 1.8, None, 1.65, 1.75],
    'Weight': [65, 80, 70, None, 85],
    'City': ['NY', 'LA', 'NY', 'SF', 'LA'],
    'Score': [None, 65, 70, 80, 85]
})

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Height'].fillna(data['Height'].mean(), inplace=True)
data['Weight'].fillna(data['Weight'].mean(), inplace=True)
data['Score'].fillna(0, inplace=True)

# One-hot encode categorical data
data = pd.get_dummies(data, columns=['City'])

# Scale data
scaler = StandardScaler()
data[['Age', 'Height', 'Weight', 'Score']] = scaler.fit_transform(data[['Age', 'Height', 'Weight', 'Score']])

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(train_data)
print(test_data)
```

This example combines filling missing values, encoding categorical data, scaling, and data splitting.

---

## 🚀 Summary

Data is the **backbone of AI**. Mastering data preprocessing - such as cleaning, encoding, and scaling - will prepare you to handle real-world data and build effective AI models.

### Next Steps:
In the next lesson, you’ll start applying data knowledge to **Machine Learning**, where data is transformed into intelligent models!

---

> **Remember**: The quality of your AI model depends heavily on the quality of the data you use.
