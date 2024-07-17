import plotly.express as px
import pandas as pd

# 샘플 데이터 생성
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June'],
    'Sales': [150, 200, 180, 220, 250, 230]
}
df = pd.DataFrame(data)

# 인터랙티브 선 그래프 생성
fig = px.line(df, x='Month', y='Sales', title='Monthly Sales')
fig.show()