import plotly.graph_objects as go

# 샘플 데이터 정의
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=["A", "B", "C", "D", "E", "F"],
        color=["blue", "blue", "blue", "blue", "blue", "blue"]
    ),
    link=dict(
        source=[0, 1, 0, 2, 3, 3],
        target=[2, 3, 3, 4, 4, 5],
        value=[8, 4, 2, 8, 4, 2]
    )
))

# * Sanky Diagram
fig.update_layout(title_text="Sample Sankey Diagram", font_size=10)
fig.show()