import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.figure_factory as ff
import dash_table as dt

# Initialise the app
app = dash.Dash(__name__)
server = app.server

# Reading Data
df_southamerica_results = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/southamerica_results.csv')

# Processing
df_pie_southamerica = df_southamerica_results.loc[:, ['clusters', 'track_id']].groupby(by='clusters', as_index=False).count()

# Visualizing
# V1
labels = ['Cluster 0: Joyful','Cluster 1: Party','Cluster 2: Chill','Cluster 3: Romantic']
fig = go.Figure(
        data=[go.Pie(labels=labels, values=df_pie_southamerica.loc[:, 'track_id'])]
      )
fig.update_layout(title_text='Moods in South America')
fig.update_layout({
  'plot_bgcolor': 'rgba(0, 0, 0, 0)',
  'paper_bgcolor': 'rgba(0, 0, 0, 0)'
})

app.layout = html.Div(children=[
    html.H1(children='Spotify Tracks Analysis in Lockdown', style={'textAlign': 'center'}),
    html.Div(children='Students: Ronie Arauco & Handry Guillen', style={'textAlign': 'center'}),
    dcc.Graph(
        id='pie-chart-southamerica',
        figure=fig
    )
])


# Graphic 1
#labels = ['Cluster 0: Joyful','Cluster 1: Party','Cluster 2: Chill','Cluster 3: Romantic']
#fig = go.Figure(
#        data=[go.Pie(labels=labels, values=df_pie_southamerica.loc[:, 'track_id'])]
#      )
#fig.update_layout(title_text='Moods in South America')
#fig.show()


if __name__ == "__main__":
    app.run_server(debug=True)
