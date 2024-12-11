#Importation of libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Task 1: Loading and Explorong the Dataset

# Loading the Iris dataset from sklearn
iris = load_iris()

# Converting the dataset to a pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Adding the target variable (species)
df['species'] = iris.target

# Mapping the target variable to species names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Displaying the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Exploring the structure of the dataset
print("\nDataset Info (Data types and missing values):")
df.info()

# Checking for missing values
print("\nMissing Values in the Dataset:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis

# Computing basic statistics for numerical columns
print("\nBasic Statistics for Numerical Columns:")
print(df.describe())

# Performing groupings on the 'species' column and computing mean of numerical columns for each group
df_grouped = df.groupby('species').mean()

print("\nMean of Numerical Columns by Species:")
print(df_grouped)

# Task 3: Data Visualization

# Visualization 1: Line chart showing trends over time (simulated as time series data)
# simulate a simple "time" progression for the average petal length
time = [1, 2, 3]  # The simulated time points
avg_petal_length = df.groupby('species')['petal length (cm)'].mean().values  # Average petal length for each species

plt.figure(figsize=(8, 6))
plt.plot(time, avg_petal_length, marker='o')
plt.title('Simulated Time Trends of Average Petal Length')
plt.xlabel('Time')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(time, ['Setosa', 'Versicolor', 'Virginica'])  # Label each point with species
plt.grid(True)
plt.show()

# Visualization 2: Bar chart comparing average petal length across species
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Visualization 3: Histogram for the distribution of sepal length
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal length (cm)'], kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Visualization 4: Scatter plot for Sepal Length vs Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Error Handling 
try:
    pass  # just a placeholder
except FileNotFoundError:
    print("File not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("The file is empty.")
except Exception as e:
    print(f"An error occurred: {e}")
