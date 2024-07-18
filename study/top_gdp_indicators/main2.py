import pandas as pd
import plotly.express as px
import seaborn as sns

file_path = 'Economic_Indicators_Top_15_GDP_Countries.csv'
df = pd.read_csv(file_path)

df['2022 [YR2022]'] = pd.to_numeric(df['2022 [YR2022]'], errors='coerce')
df['2023 [YR2023]'] = pd.to_numeric(df['2023 [YR2023]'], errors='coerce')

numeric_columns = ['2019 [YR2019]', '2020 [YR2020]', '2021 [YR2021]', '2022 [YR2022]', '2023 [YR2023]']
df_numeric = df[['Country Name', 'Series Name'] + numeric_columns]

df_melted = df_numeric.melt(id_vars=['Country Name', 'Series Name'], var_name='Year', value_name='Value')

df_melted['Year'] = df_melted['Year'].apply(lambda x: int(x.split()[0]))

series_names = [
    "GDP (current US$)",
    "GDP (constant 2015 US$)",
    "GDP growth (annual %)",
    "GDP per capita (current US$)",
    "GDP per capita (constant 2015 US$)",
    "GDP per capita growth (annual %)",
    "Inflation, GDP deflator (annual %)",
    "Inflation, consumer prices (annual %)",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)",
    "Unemployment, youth total (% of total labor force ages 15-24) (national estimate)",
    "External balance on goods and services (current US$)",
]

sns.set(style="whitegrid")

for series in series_names:
    series_data = df_melted[df_melted['Series Name'] == series]

    sorted_series_data = series_data.sort_values(by=['Year', 'Value'], ascending=[True, False])
    
    fig = px.line(
        sorted_series_data,
        x='Year',
        y='Value',
        color='Country Name',
        markers=True,
        title=f'{series} Over Years for Top 15 GDP Countries',
    )
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=series,
        legend_title='Country',
        hovermode='x unified'
    )

    fig.show()

