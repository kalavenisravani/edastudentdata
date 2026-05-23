import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/tudent_dataset.csv")

# First 5 rows
print("FIRST 5 ROWS")
print(df.head())

# Shape
print("\nSHAPE")
print(df.shape)

# Columns
print("\nCOLUMNS")
print(df.columns)

# Info
print("\nINFO")
print(df.info())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Statistical summary
print("\nSTATISTICS")
print(df.describe())

# Histogram
df.hist(figsize=(12,10))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Scatter Plot
sns.scatterplot(x='math score', y='reading score', data=df)
plt.title("Math vs Reading Scores")
plt.show()

# Box Plot
sns.boxplot(x=df['writing score'])
plt.title("Writing Score Distribution")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY")