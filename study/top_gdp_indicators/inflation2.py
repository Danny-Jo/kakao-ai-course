import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Economic_Indicators_Top_15_GDP_Countries.csv'
df = pd.read_csv(file_path)

inflation_columns = ['Country Name', 'Country Code', '2019 [YR2019]', '2020 [YR2020]', '2021 [YR2021]', '2022 [YR2022]', '2023 [YR2023]']

unique_countries = df['Country Name'].unique()
palette = dict(zip(unique_countries, sns.color_palette("husl", len(unique_countries))))

df_inflation_gdp_deflator = df[df['Series Name'] == 'Inflation, GDP deflator (annual %)'][inflation_columns]

for col in inflation_columns[2:]:
    df_inflation_gdp_deflator[col] = pd.to_numeric(df_inflation_gdp_deflator[col], errors='coerce')

df_inflation_gdp_deflator['Average Inflation'] = df_inflation_gdp_deflator[inflation_columns[2:]].mean(axis=1)

df_inflation_gdp_deflator = df_inflation_gdp_deflator.sort_values(by='Average Inflation', ascending=False)

df_inflation_gdp_deflator['Country Name (Avg)'] = df_inflation_gdp_deflator['Country Name'] + ' (' + df_inflation_gdp_deflator['Average Inflation'].round(2).astype(str) + '%)'

df_inflation_gdp_deflator['Color'] = df_inflation_gdp_deflator['Country Name'].map(palette)
new_palette_gdp_deflator = dict(zip(df_inflation_gdp_deflator['Country Name (Avg)'], df_inflation_gdp_deflator['Color']))

plt.figure(figsize=(16, 10))
bars1 = sns.barplot(x='Country Name (Avg)', y='Average Inflation', data=df_inflation_gdp_deflator, palette=new_palette_gdp_deflator)
plt.title('Average Inflation, GDP Deflator (annual %) by Country')
plt.xlabel('Country')
plt.ylabel('Average Inflation, GDP Deflator (annual %)')
plt.xticks(rotation=45, fontsize=8)  

for bar in bars1.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

df_inflation_consumer_prices = df[df['Series Name'] == 'Inflation, consumer prices (annual %)'][inflation_columns]

for col in inflation_columns[2:]:
    df_inflation_consumer_prices[col] = pd.to_numeric(df_inflation_consumer_prices[col], errors='coerce')

df_inflation_consumer_prices['Average Inflation'] = df_inflation_consumer_prices[inflation_columns[2:]].mean(axis=1)

df_inflation_consumer_prices = df_inflation_consumer_prices.sort_values(by='Average Inflation', ascending=False)

df_inflation_consumer_prices['Country Name (Avg)'] = df_inflation_consumer_prices['Country Name'] + ' (' + df_inflation_consumer_prices['Average Inflation'].round(2).astype(str) + '%)'

df_inflation_consumer_prices['Color'] = df_inflation_consumer_prices['Country Name'].map(palette)
new_palette_consumer_prices = dict(zip(df_inflation_consumer_prices['Country Name (Avg)'], df_inflation_consumer_prices['Color']))

plt.figure(figsize=(16, 10))
bars2 = sns.barplot(x='Country Name (Avg)', y='Average Inflation', data=df_inflation_consumer_prices, palette=new_palette_consumer_prices)
plt.title('Average Inflation, Consumer Prices (annual %) by Country')
plt.xlabel('Country')
plt.ylabel('Average Inflation, Consumer Prices (annual %)')
plt.xticks(rotation=45, fontsize=8) 

for bar in bars2.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
