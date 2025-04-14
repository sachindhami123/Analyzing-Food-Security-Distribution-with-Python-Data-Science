Import data:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\deeps\Downloads\PythonExcellProject(1)\ZIP\7115\7115_source_data.csv", encoding="unicode_escape")
print(df)
print(df.head())
df.shape
df.head(5)
df.info()


Drop null value

df.dropna(inplace=True)
df.shape



Drop 0 value

df.replace(0, np.nan, inplace=True) 
df.dropna(inplace=True)  
df.shape



Check for missing values

print("Missing values in dataset:")
print(df.isnull().sum())




To change the data type of a perticular column

df['Manual distribution of food grains'].dtypes
for col in df.columns:
    if df[col].dtype == 'object':  
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
df["Manual distribution of food grains"] = df["Manual distribution of food grains"].astype('int')
df['Manual distribution of food grains'].dtypes



To rename the column name

df.rename(columns={"Manual distribution of food grains":"Direct Food Distribution"})



Check for duplicate rows

print("Duplicate rows:", df.duplicated().sum())



Remove duplicates if any

df = df.drop_duplicates()
df.shape



Final data type after cleaning

print(df.info())
print(df.head())
df.describe()



Exploratory Data Analysis:


line plot for food grains allocated over time using the Year column(how food grain allocations have changed over time.

plt.figure(figsize=(7, 3))
sns.lineplot(x=df["YearCode"], y=df["Food grains allocated"], marker="o", color="red")
plt.title("Trend of Food Grains Allocation Over Time")
plt.xlabel("Year")
plt.ylabel("Food Grains Allocated")
plt.grid(True)


Histogram of Food Grains Allocated

plt.figure(figsize=(7, 3))
sns.histplot(df["Food grains allocated"], bins=30, kde=True, color="green", label="Food Grains Distribution")
plt.title("Distribution of Food Grains Allocated")
plt.xlabel("Food Grains Allocated")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
# This will help us understand how food grains are distributed—whether it's normally distributed or skewed.



 Boxplot to Identify Outliers

plt.figure(figsize=(7, 3))
sns.boxplot(x=df["Food grains allocated"], color="blue", label="Food Grains Boxplot")
plt.title("Boxplot of Food Grains Allocated")
plt.xlabel("Food Grains Allocated")
plt.legend()
plt.grid(True)
# This will help us identify states/months where food grain allocation is extremely high or low.



Trend of Food Grains Allocation Over Time

plt.figure(figsize=(6, 3))
df_grouped = df.groupby("Year")["Food grains allocated"].sum()
df_grouped.plot(marker="o", linestyle="-", color="green", label="Total Food Grains Allocated")
plt.title("Total Food Grains Allocated Over Time")
plt.xlabel("Year")
plt.ylabel("Total Food Grains Allocated")
plt.legend()
plt.grid(True)
# This shows how food allocation changed over the years, indicating trends or policy impacts.



Aadhaar Transactions by State (Top 10 States)

plt.figure(figsize=(7, 3))
top_states = df.groupby("srcStateName")["Aadhaar authenticated Transactions"].sum().nlargest(10)
top_states.plot(kind="bar", color="yellow", label="Aadhaar Transactions")
plt.title("Top 10 States by Aadhaar Authenticated Transactions")
plt.xlabel("State")
plt.ylabel("Total Aadhaar Transactions")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)



Scatter Plot: Food Grains vs Aadhaar Transactions

plt.figure(figsize=(7, 3))
sns.scatterplot(x=df["Food grains allocated"], y=df["Aadhaar authenticated Transactions"], color="blue", alpha=0.6, label="Data Points")
plt.title("Food Grains Allocated vs Aadhaar Authenticated Transactions")
plt.xlabel("Food Grains Allocated")
plt.ylabel("Aadhaar Authenticated Transactions")
plt.legend()
plt.grid(True)
# This will help us see if higher food grain allocation results in higher Aadhaar authenticated transactions.




Pie Chart for Distribution of Transactions

plt.figure(figsize=(7, 3))
labels = ["ePoS Distribution", "Manual Distribution"]
sizes = [df["ePoS (Electronic Point of Sale system) distribution of food grains"].sum(), df["Manual distribution of food grains"].sum()]
colors = ["skyblue", "lightcoral"]
plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
plt.title("Distribution of Food Grains: ePoS vs Manual")
plt.legend(labels, title="Legend")
plt.show()
# This gives a clear picture of how much food grain is distributed electronically vs manually.




Correlation Matrix of Numeric Features (Heatmap)

# Correlation helps understand relationships between numeric variables
numeric_df = df.select_dtypes(include=['number'])
# heatmap
plt.figure(figsize=(7, 3))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()



efficient way to plot histograms for all numeric columns in one go

numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols].hist(figsize=(13, 10), bins=30, color="blue", edgecolor="black", layout=(len(numeric_cols) // 3 + 1, 3))
plt.suptitle("Distribution of Numeric Columns")
plt.show()



Boxplots for Outlier Detection

plt.figure(figsize=(7, 3))
df[numeric_cols].boxplot(rot=90)
plt.title("Boxplot of Numeric Columns")
plt.show()


Time-Based Analysis


# Convert 'Month' to datetime format
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
# Extract the year from 'Month'
df['Year'] = df['Month'].dt.year
# Group by 'Year' and sum only numeric columns
df.groupby('Year').sum(numeric_only=True).plot(kind='bar', figsize=(9, 4))
plt.title("Yearly Trends of Numeric Features")
plt.xlabel("Year")
plt.ylabel("Sum of Numeric Features")
plt.show()
