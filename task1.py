
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = sns.load_dataset('iris')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Distribution of numerical features
plt.figure(figsize=(10, 6))
df.hist(bins=20, edgecolor='black', figsize=(10, 6))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(df, hue='species')
plt.suptitle('Pairplot of the Iris Dataset', y=1.02)
plt.show()

# Correlation matrix
corr_matrix = df.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Boxplot to identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title('Boxplot of Numerical Features')
plt.show()

# Scatter plot to check relationships
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species')
plt.title('Scatter Plot of Sepal Length vs Petal Length')
plt.show()

# Display the correlation matrix values
print("\nCorrelation matrix:")
print(corr_matrix)

# Display the boxplot statistics
print("\nBoxplot statistics:")
for col in df.columns[:-1]:  # Exclude 'species' column
    print(f"\n{col} statistics:")
    print(df[col].describe())

# Identify and display potential outliers
def find_outliers(column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

print("\nPotential outliers:")
for col in df.columns[:-1]:  # Exclude 'species' column
    outliers = find_outliers(col)
    if not outliers.empty:
        print(f"\nOutliers in {col}:")
        print(outliers)

# End of EDA code