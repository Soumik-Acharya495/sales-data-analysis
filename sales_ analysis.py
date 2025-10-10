# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
python

# Step 2: Load Dataset
df = pd.read_csv("sales_data.csv")
df.head()
python
Copy code
# Step 3: Basic Data Exploration
print("Dataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())
python
Copy code
# Step 4: Data Cleaning
df.fillna(0, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
python
Copy code
# Step 5.1: Total Sales by Region
region_sales = df.groupby('Region')['Total_Sales'].sum().sort_values(ascending=False)
region_sales.plot(kind='bar', color='skyblue')
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.show()
python
Copy code
# Step 5.2: Top 5 Cities by Sales
top_cities = df.groupby('City')['Total_Sales'].sum().nlargest(5)
sns.barplot(x=top_cities.values, y=top_cities.index, palette='viridis')
plt.title("Top 5 Cities by Sales")
plt.xlabel("Total Sales")
plt.ylabel("City")
plt.show()
python
Copy code
# Step 5.3: Monthly Sales Trend
monthly_sales = df.groupby('Month')['Total_Sales'].sum()
monthly_sales.plot(marker='o', color='green')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
python
Copy code
# Step 5.4: Product-wise Sales Performance
product_sales = df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False)
sns.barplot(x=product_sales.values, y=product_sales.index, palette='coolwarm')
plt.title("Product-wise Sales Performance")
plt.xlabel("Total Sales")
plt.ylabel("Product")
plt.show()
python
Copy code
# Step 5.5: Correlation Heatmap
sns.heatmap(df[['Quantity', 'Unit_Price', 'Total_Sales']].corr(), annot=True, cmap='Blues')
plt.title("Correlation Heatmap")
plt.show()
python
Copy code
# Step 6: Predict Next Month's Sales (Optional)
from sklearn.linear_model import LinearRegression

X = np.arange(len(monthly_sales)).reshape(-1, 1)
y = monthly_sales.values

model = LinearRegression()
model.fit(X, y)

predicted = model.predict([[len(X)]])
print("Predicted Next Month Sales:", predicted[0])
