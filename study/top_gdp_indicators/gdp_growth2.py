import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
file_path = 'Economic_Indicators_Top_15_GDP_Countries.csv'
df = pd.read_csv(file_path)

# Select only the necessary columns
growth_columns = ['Country Name', 'Country Code', '2019 [YR2019]', '2020 [YR2020]', '2021 [YR2021]', '2022 [YR2022]', '2023 [YR2023]']

# Select GDP growth (annual %) data
df_gdp_growth = df[df['Series Name'] == 'GDP growth (annual %)'][growth_columns]

# Convert data types to float
for col in growth_columns[2:]:
    df_gdp_growth[col] = pd.to_numeric(df_gdp_growth[col], errors='coerce')

# Select GDP per capita growth (annual %) data
df_gdp_per_capita_growth = df[df['Series Name'] == 'GDP per capita growth (annual %)'][growth_columns]

# Convert data types to float
for col in growth_columns[2:]:
    df_gdp_per_capita_growth[col] = pd.to_numeric(df_gdp_per_capita_growth[col], errors='coerce')

# Calculate average GDP growth for each country
df_gdp_growth['Average GDP Growth'] = df_gdp_growth[growth_columns[2:]].mean(axis=1)

# Calculate average GDP per capita growth for each country
df_gdp_per_capita_growth['Average GDP per Capita Growth'] = df_gdp_per_capita_growth[growth_columns[2:]].mean(axis=1)

# Sort data
df_gdp_growth = df_gdp_growth.sort_values(by='Average GDP Growth', ascending=False)
df_gdp_per_capita_growth = df_gdp_per_capita_growth.sort_values(by='Average GDP per Capita Growth', ascending=False)

# Plot average GDP growth bar chart
plt.figure(figsize=(16, 10))
sns.barplot(x='Country Name', y='Average GDP Growth', data=df_gdp_growth, palette='viridis')
plt.title('Average GDP Growth (Annual %) Comparison by Country')
plt.xlabel('Country')
plt.ylabel('Average GDP Growth (Annual %)')
plt.xticks(rotation=45, fontsize=10)
plt.show()

# Plot average GDP per capita growth bar chart
plt.figure(figsize=(16, 10))
sns.barplot(x='Country Name', y='Average GDP per Capita Growth', data=df_gdp_per_capita_growth, palette='viridis')
plt.title('Average GDP per Capita Growth (Annual %) Comparison by Country')
plt.xlabel('Country')
plt.ylabel('Average GDP per Capita Growth (Annual %)')
plt.xticks(rotation=45, fontsize=10)
plt.show()
