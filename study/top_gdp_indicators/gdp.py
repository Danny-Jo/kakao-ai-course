import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Economic_Indicators_Top_15_GDP_Countries.csv'
df = pd.read_csv(file_path)

gdp_columns = ['Country Name', 'Country Code', '2019 [YR2019]', '2020 [YR2020]', '2021 [YR2021]', '2022 [YR2022]', '2023 [YR2023]']

df_gdp_current = df[df['Series Name'] == 'GDP (current US$)'][gdp_columns]

for col in gdp_columns[2:]:
    df_gdp_current[col] = pd.to_numeric(df_gdp_current[col], errors='coerce')

df_gdp_current = df_gdp_current.sort_values(by='2023 [YR2023]', ascending=False)

df_gdp_current['Rank'] = range(1, len(df_gdp_current) + 1)
df_gdp_current['Country Name'] = df_gdp_current['Rank'].astype(str) + ' - ' + df_gdp_current['Country Name']

df_melted_current = df_gdp_current.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='GDP')

df_melted_current['Country Name'] = pd.Categorical(df_melted_current['Country Name'], categories=df_gdp_current['Country Name'], ordered=True)

df_gdp_constant = df[df['Series Name'] == 'GDP (constant 2015 US$)'][gdp_columns]

for col in gdp_columns[2:]:
    df_gdp_constant[col] = pd.to_numeric(df_gdp_constant[col], errors='coerce')

df_gdp_constant = df_gdp_constant.sort_values(by='2023 [YR2023]', ascending=False)

df_gdp_constant['Rank'] = range(1, len(df_gdp_constant) + 1)
df_gdp_constant['Country Name'] = df_gdp_constant['Rank'].astype(str) + ' - ' + df_gdp_constant['Country Name']

df_melted_constant = df_gdp_constant.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='GDP')

df_melted_constant['Country Name'] = pd.Categorical(df_melted_constant['Country Name'], categories=df_gdp_constant['Country Name'], ordered=True)

plt.figure(figsize=(16, 10))
sns.barplot(x='Country Name', y='GDP', hue='Year', data=df_melted_current)
plt.title('GDP (current US$) Comparison by Country and Year')
plt.xlabel('Country')
plt.ylabel('GDP (current US$)')
plt.xticks(rotation=45, fontsize=10) 
plt.legend(title='Year')
plt.show()

plt.figure(figsize=(16, 10))
sns.barplot(x='Country Name', y='GDP', hue='Year', data=df_melted_constant)
plt.title('GDP (constant 2015 US$) Comparison by Country and Year')
plt.xlabel('Country')
plt.ylabel('GDP (constant 2015 US$)')
plt.xticks(rotation=45, fontsize=10)  
plt.legend(title='Year')
plt.show()
