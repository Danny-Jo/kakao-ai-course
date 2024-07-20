import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Economic_Indicators_Top_15_GDP_Countries.csv'
df = pd.read_csv(file_path)

inflation_columns = ['Country Name', 'Country Code', '2019 [YR2019]', '2020 [YR2020]', '2021 [YR2021]', '2022 [YR2022]', '2023 [YR2023]']

df_inflation_gdp_deflator = df[df['Series Name'] == 'Inflation, GDP deflator (annual %)'][inflation_columns]

for col in inflation_columns[2:]:
    df_inflation_gdp_deflator[col] = pd.to_numeric(df_inflation_gdp_deflator[col], errors='coerce')

df_inflation_gdp_deflator['Average Inflation'] = df_inflation_gdp_deflator[inflation_columns[2:]].mean(axis=1)

df_inflation_gdp_deflator['Country Name'] = df_inflation_gdp_deflator['Country Name'] + ' (' + df_inflation_gdp_deflator['Average Inflation'].round(2).astype(str) + '%)'

df_inflation_gdp_deflator = df_inflation_gdp_deflator.sort_values(by='Average Inflation', ascending=False)

df_melted_inflation_gdp_deflator = df_inflation_gdp_deflator.melt(id_vars=['Country Name', 'Country Code', 'Average Inflation'], var_name='Year', value_name='Inflation')

df_melted_inflation_gdp_deflator['Country Name'] = pd.Categorical(df_melted_inflation_gdp_deflator['Country Name'], categories=df_inflation_gdp_deflator['Country Name'], ordered=True)

df_inflation_consumer_prices = df[df['Series Name'] == 'Inflation, consumer prices (annual %)'][inflation_columns]

for col in inflation_columns[2:]:
    df_inflation_consumer_prices[col] = pd.to_numeric(df_inflation_consumer_prices[col], errors='coerce')

df_inflation_consumer_prices['Average Inflation'] = df_inflation_consumer_prices[inflation_columns[2:]].mean(axis=1)

df_inflation_consumer_prices['Country Name'] = df_inflation_consumer_prices['Country Name'] + ' (' + df_inflation_consumer_prices['Average Inflation'].round(2).astype(str) + '%)'

df_inflation_consumer_prices = df_inflation_consumer_prices.sort_values(by='Average Inflation', ascending=False)

df_melted_inflation_consumer_prices = df_inflation_consumer_prices.melt(id_vars=['Country Name', 'Country Code', 'Average Inflation'], var_name='Year', value_name='Inflation')

df_melted_inflation_consumer_prices['Country Name'] = pd.Categorical(df_melted_inflation_consumer_prices['Country Name'], categories=df_inflation_consumer_prices['Country Name'], ordered=True)

plt.figure(figsize=(16, 10))
sns.barplot(x='Country Name', y='Inflation', hue='Year', data=df_melted_inflation_gdp_deflator)
plt.title('Inflation, GDP Deflator (annual %) Comparison by Country and Year')
plt.xlabel('Country')
plt.ylabel('Inflation, GDP Deflator (annual %)')
plt.xticks(rotation=45, fontsize=8)  
plt.legend(title='Year')
plt.show()

plt.figure(figsize=(16, 10))
sns.barplot(x='Country Name', y='Inflation', hue='Year', data=df_melted_inflation_consumer_prices)
plt.title('Inflation, Consumer Prices (annual %) Comparison by Country and Year')
plt.xlabel('Country')
plt.ylabel('Inflation, Consumer Prices (annual %)')
plt.xticks(rotation=45, fontsize=8)  
plt.legend(title='Year')
plt.show()
