import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
%matplotlib inline
df = pd.read_csv("covid_data.csv")
df.head()
python
Copy code
df.info()
df.describe()
df.isnull().sum()
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
python
Copy code
total_cases = df.groupby('country')['total_cases'].max().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=total_cases.values, y=total_cases.index, palette='Reds_r')
plt.title("Top 10 Countries by Total COVID-19 Cases")
plt.xlabel("Total Cases")
plt.ylabel("Country")
plt.show()
total_deaths = df.groupby('country')['total_deaths'].max().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=total_deaths.values, y=total_deaths.index, palette='Greys')
plt.title("Top 10 Countries by Total COVID-19 Deaths")
plt.xlabel("Total Deaths")
plt.ylabel("Country")
plt.show()
monthly_cases = df.groupby('month')['new_cases'].sum()
plt.figure(figsize=(10,5))
plt.plot(monthly_cases.index.astype(str), monthly_cases.values, marker='o', color='blue')
plt.title("Monthly COVID-19 Cases Trend")
plt.xlabel("Month")
plt.ylabel("New Cases")
plt.xticks(rotation=45)
plt.show()
monthly_deaths = df.groupby('month')['new_deaths'].sum()
plt.figure(figsize=(10,5))
plt.plot(monthly_deaths.index.astype(str), monthly_deaths.values, marker='o', color='red')
plt.title("Monthly COVID-19 Deaths Trend")
plt.xlabel("Month")
plt.ylabel("New Deaths")
plt.xticks(rotation=45)
plt.show()
correlation = df[['total_cases', 'total_deaths', 'population']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap - COVID-19 Data")
plt.show()
fig = px.scatter(df, x='total_cases', y='total_deaths', color='continent',
                 size='population', hover_name='country', log_x=True, log_y=True,
                 title='COVID-19 Total Cases vs Deaths (Log Scale)')
fig.show()
python
Copy code
fig = px.choropleth(df, locations='country', locationmode='country names',
                    color='total_cases', hover_name='country',
                    color_continuous_scale='Reds', title='Global COVID-19 Cases Map')
fig.show()
python
Copy code
highest_recovery = df.groupby('country')['total_recovered'].max().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=highest_recovery.values, y=highest_recovery.index, palette='Greens')
plt.title("Top 10 Countries by Recoveries")
plt.xlabel("Total Recovered")
plt.ylabel("Country")
plt.show()
python
Copy code
death_rate = (df['total_deaths'].sum() / df['total_cases'].sum()) * 100
recovery_rate = (df['total_recovered'].sum() / df['total_cases'].sum()) * 100
print("Global Death Rate: {:.2f}%".format(death_rate))
print("Global Recovery Rate: {:.2f}%".format(recovery_rate))






