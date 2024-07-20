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

# Sort and prepare GDP growth (annual %) data for visualization
df_gdp_growth = df_gdp_growth.sort_values(by='2023 [YR2023]', ascending=False)
df_melted_growth = df_gdp_growth.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='GDP Growth')

# Set the order of Country Name
df_melted_growth['Country Name'] = pd.Categorical(df_melted_growth['Country Name'], categories=df_gdp_growth['Country Name'], ordered=True)

# Sort and prepare GDP per capita growth (annual %) data for visualization
df_gdp_per_capita_growth = df_gdp_per_capita_growth.sort_values(by='2023 [YR2023]', ascending=False)
df_melted_per_capita_growth = df_gdp_per_capita_growth.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='GDP per Capita Growth')

# Set the order of Country Name
df_melted_per_capita_growth['Country Name'] = pd.Categorical(df_melted_per_capita_growth['Country Name'], ordered=True)

# Plot GDP growth (annual %) bar chart
plt.figure(figsize=(16, 10))
sns.barplot(x='Country Name', y='GDP Growth', hue='Year', data=df_melted_growth)
plt.title('GDP Growth (Annual %) Comparison by Country and Year')
plt.xlabel('Country')
plt.ylabel('GDP Growth (Annual %)')
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='Year')
plt.show()

# Plot GDP per capita growth (annual %) bar chart
plt.figure(figsize=(16, 10))
sns.barplot(x='Country Name', y='GDP per Capita Growth', hue='Year', data=df_melted_per_capita_growth)
plt.title('GDP per Capita Growth (Annual %) Comparison by Country and Year')
plt.xlabel('Country')
plt.ylabel('GDP per Capita Growth (Annual %)')
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='Year')
plt.show()
