import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("titanic.csv")

# Basic information
print("Shape:")
print(df.shape)

print("\nInfo:")
df.info()

print("\nStatistics:")
print(df.describe())

print("\nFirst 10 rows:")
print(df.head(10))

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

print("\nMissing percentage:")
print((df.isnull().mean() * 100).round(2))

# Check duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# Histograms
cols = ["Age", "Fare", "SibSp", "Parch"]

for col in cols:
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(col + " Distribution")
    plt.show()

# Count plots
print(df["Sex"].value_counts())
print(df["Pclass"].value_counts())
print(df["Embarked"].value_counts())

sns.countplot(x="Survived", data=df)
plt.title("Survived")
plt.show()

# Correlation
corr = df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Boxplots
sns.boxplot(x="Pclass", y="Age", data=df)
plt.title("Age vs Passenger Class")
plt.show()

sns.boxplot(x="Survived", y="Fare", data=df)
plt.title("Fare vs Survival")
plt.show()

# Cleaning
# Fill missing age with median of each class
df["Age"] = df["Age"].fillna(
    df.groupby("Pclass")["Age"].transform("median")
)

# Fill missing embarked values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove Cabin because most values are missing
df.drop("Cabin", axis=1, inplace=True)

# Remove columns we don't need
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nData info after cleaning:")
df.info()

# Outliers
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower = max(0, Q1 - 1.5 * IQR)
upper = Q3 + 1.5 * IQR

print("Fare limits:", lower, upper)

outliers = df[(df["Fare"] < lower) | (df["Fare"] > upper)]
print("Number of outliers:", len(outliers))

sns.boxplot(x=df["Fare"])
plt.title("Fare Boxplot")
plt.show()

# More plots
plt.scatter(df["Age"], df["Fare"], alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare")
plt.show()

sns.pairplot(df[["Age", "Fare", "Pclass", "Survived"]])
plt.show()

corr = df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation After Cleaning")
plt.show()

print("\nFinal shape:")
print(df.shape)