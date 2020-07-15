import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.figure_factory as ff
import dash_table as dt
from dash.dependencies import Input, Output

# Initialise the app
app = dash.Dash(__name__)
server = app.server

# Reading Data
df_southamerica_results = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/southamerica_results.csv')
df_southamerica = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/southamerica_history.csv')
df_northamerica_results = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/northamerica_results.csv')
df_northamerica = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/northamerica_history.csv')
df_europe_results = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/europe_results.csv')
df_europe = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/europe_history.csv')
df_asia_results = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/asia_results.csv')
df_asia = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/asia_history.csv')
df_africa_results = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/africa_results.csv')
df_africa = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/africa_history.csv')
df_oceania_results = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/oceania_results.csv')
df_oceania = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/oceania_history.csv')
df_peru_results = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/peru_results.csv')
df_peru = pd.read_csv('https://dl-ta-dataset.s3.amazonaws.com/peru_history.csv')

# Processing
df_pie_southamerica = df_southamerica_results.loc[:, ['clusters', 'track_id']].groupby(by='clusters', as_index=False).count()
df_pie_peru = df_peru_results.loc[:, ['clusters', 'track_id']].groupby(by='clusters', as_index=False).count()
df_pie_northamerica = df_northamerica_results.loc[:, ['clusters', 'track_id']].groupby(by='clusters', as_index=False).count()
df_pie_europe = df_europe_results.loc[:, ['clusters', 'track_id']].groupby(by='clusters', as_index=False).count()
df_pie_asia = df_asia_results.loc[:, ['clusters', 'track_id']].groupby(by='clusters', as_index=False).count()
df_pie_africa = df_africa_results.loc[:, ['clusters', 'track_id']].groupby(by='clusters', as_index=False).count()
df_pie_oceania = df_oceania_results.loc[:, ['clusters', 'track_id']].groupby(by='clusters', as_index=False).count()

# Valence
df_peru_valence = df_peru.loc[:, ['date_extraction', 'track_valence']]
df_peru_valence.loc[:, 'date_extraction'] = df_peru_valence.loc[:, 'date_extraction'].astype('datetime64[ns]')
df_peru_valence_weekly = df_peru_valence.resample('W-Mon', closed='right', label='right', on='date_extraction').mean()
df_peru_valence_weekly = df_peru_valence_weekly.reset_index()
fig_valence_peru = px.line(df_peru_valence_weekly, x='date_extraction', y='track_valence', title='Valence over time')
fig_valence_peru.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
df_southamerica_valence = df_southamerica.loc[:, ['date_extraction', 'track_valence']]
df_southamerica_valence.loc[:, 'date_extraction'] = df_southamerica_valence.loc[:, 'date_extraction'].astype('datetime64[ns]')
df_southamerica_valence_weekly = df_southamerica_valence.resample('W-Mon', closed='right', label='right', on='date_extraction').mean()
df_southamerica_valence_weekly = df_southamerica_valence_weekly.reset_index()
fig_valence_southamerica = px.line(df_southamerica_valence_weekly, x='date_extraction', y='track_valence', title='Valence over time')
fig_valence_southamerica.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
df_northamerica_valence = df_northamerica.loc[:, ['date_extraction', 'track_valence']]
df_northamerica_valence.loc[:, 'date_extraction'] = df_northamerica_valence.loc[:, 'date_extraction'].astype('datetime64[ns]')
df_northamerica_valence_weekly = df_northamerica_valence.resample('W-Mon', closed='right', label='right', on='date_extraction').mean()
df_northamerica_valence_weekly = df_northamerica_valence_weekly.reset_index()
fig_valence_northamerica = px.line(df_northamerica_valence_weekly, x='date_extraction', y='track_valence', title='Valence over time')
fig_valence_northamerica.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

df_europe_valence = df_europe.loc[:, ['date_extraction', 'track_valence']]
df_europe_valence.loc[:, 'date_extraction'] = df_europe_valence.loc[:, 'date_extraction'].astype('datetime64[ns]')
df_europe_valence_weekly = df_europe_valence.resample('W-Mon', closed='right', label='right', on='date_extraction').mean()
df_europe_valence_weekly = df_europe_valence_weekly.reset_index()
fig_valence_europe = px.line(df_europe_valence_weekly, x='date_extraction', y='track_valence', title='Valence over time')
fig_valence_europe.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

df_asia_valence = df_asia.loc[:, ['date_extraction', 'track_valence']]
df_asia_valence.loc[:, 'date_extraction'] = df_asia_valence.loc[:, 'date_extraction'].astype('datetime64[ns]')
df_asia_valence_weekly = df_asia_valence.resample('W-Mon', closed='right', label='right', on='date_extraction').mean()
df_asia_valence_weekly = df_asia_valence_weekly.reset_index()
fig_valence_asia = px.line(df_asia_valence_weekly, x='date_extraction', y='track_valence', title='Valence over time')
fig_valence_asia.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

df_africa_valence = df_africa.loc[:, ['date_extraction', 'track_valence']]
df_africa_valence.loc[:, 'date_extraction'] = df_africa_valence.loc[:, 'date_extraction'].astype('datetime64[ns]')
df_africa_valence_weekly = df_africa_valence.resample('W-Mon', closed='right', label='right', on='date_extraction').mean()
df_africa_valence_weekly = df_africa_valence_weekly.reset_index()
fig_valence_africa = px.line(df_africa_valence_weekly, x='date_extraction', y='track_valence', title='Valence over time')
fig_valence_africa.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

df_oceania_valence = df_oceania.loc[:, ['date_extraction', 'track_valence']]
df_oceania_valence.loc[:, 'date_extraction'] = df_oceania_valence.loc[:, 'date_extraction'].astype('datetime64[ns]')
df_oceania_valence_weekly = df_oceania_valence.resample('W-Mon', closed='right', label='right', on='date_extraction').mean()
df_oceania_valence_weekly = df_oceania_valence_weekly.reset_index()
fig_valence_oceania = px.line(df_oceania_valence_weekly, x='date_extraction', y='track_valence', title='Valence over time')
fig_valence_oceania.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# Visualizing
labels_southamerica = ['Cluster 0: Joyful','Cluster 1: Party','Cluster 2: Chill','Cluster 3: Romantic']
labels_peru = ['Cluster 0: Chill','Cluster 1: Romantic','Cluster 2: Party','Cluster 3: Joyful']
labels_unkown = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

fig_pie_southamerica = go.Figure(
        data=[go.Pie(labels=labels_southamerica, values=df_pie_southamerica.loc[:, 'track_id'])]
      )
fig_pie_peru = go.Figure(
        data=[go.Pie(labels=labels_peru, values=df_pie_peru.loc[:, 'track_id'])]
      )
fig_pie_southamerica.update_layout(title_text='Moods in South America')
fig_pie_peru.update_layout(title_text='Moods in Peru')
fig_pie_northamerica = go.Figure(
        data=[go.Pie(labels=labels_unkown, values=df_pie_northamerica.loc[:, 'track_id'])]
      )
fig_pie_northamerica.update_layout(title_text='Moods in North America')
fig_pie_europe = go.Figure(
        data=[go.Pie(labels=labels_unkown, values=df_pie_europe.loc[:, 'track_id'])]
      )
fig_pie_europe.update_layout(title_text='Moods in Europe')
fig_pie_asia = go.Figure(
        data=[go.Pie(labels=labels_unkown, values=df_pie_asia.loc[:, 'track_id'])]
      )
fig_pie_asia.update_layout(title_text='Moods in Asia')
fig_pie_africa = go.Figure(
        data=[go.Pie(labels=labels_unkown, values=df_pie_africa.loc[:, 'track_id'])]
      )
fig_pie_africa.update_layout(title_text='Moods in Africa')
fig_pie_oceania = go.Figure(
        data=[go.Pie(labels=labels_unkown, values=df_pie_oceania.loc[:, 'track_id'])]
      )
fig_pie_oceania.update_layout(title_text='Moods in Oceania')

# Tables
peru0 = df_peru_results.loc[df_peru_results.loc[:, 'clusters'] == 0]
peru1 = df_peru_results.loc[df_peru_results.loc[:, 'clusters'] == 1]
peru2 = df_peru_results.loc[df_peru_results.loc[:, 'clusters'] == 2]
peru3 = df_peru_results.loc[df_peru_results.loc[:, 'clusters'] == 3]
peru0['clusters'] = peru0['clusters'].map({ 0 : 'Chill', 1: 'Romantic', 2 : 'Party', 3 : 'Joyful'})
peru1['clusters'] = peru1['clusters'].map({ 0 : 'Chill', 1: 'Romantic', 2 : 'Party', 3 : 'Joyful'})
peru2['clusters'] = peru2['clusters'].map({ 0 : 'Chill', 1: 'Romantic', 2 : 'Party', 3 : 'Joyful'})
peru3['clusters'] = peru3['clusters'].map({ 0 : 'Chill', 1: 'Romantic', 2 : 'Party', 3 : 'Joyful'})
perusample0 = peru0.sample(10)
fig_table_peru0 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[perusample0.track_name, perusample0.artist, perusample0.album, perusample0.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_peru0.update_layout(title_text='Chill tracks in Peru')
perusample1 = peru1.sample(10)
fig_table_peru1 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[perusample1.track_name, perusample1.artist, perusample1.album, perusample1.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_peru1.update_layout(title_text='Romantic tracks in Peru')
perusample2 = peru2.sample(10)
fig_table_peru2 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[perusample2.track_name, perusample2.artist, perusample2.album, perusample2.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_peru2.update_layout(title_text='Party tracks in Peru')
perusample3 = peru3.sample(10)
fig_table_peru3 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[perusample3.track_name, perusample3.artist, perusample3.album, perusample3.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_peru3.update_layout(title_text='Joyful tracks in Peru')

southamerica0 = df_southamerica_results.loc[df_southamerica_results.loc[:, 'clusters'] == 0]
southamerica1 = df_southamerica_results.loc[df_southamerica_results.loc[:, 'clusters'] == 1]
southamerica2 = df_southamerica_results.loc[df_southamerica_results.loc[:, 'clusters'] == 2]
southamerica3 = df_southamerica_results.loc[df_southamerica_results.loc[:, 'clusters'] == 3]
southamerica0['clusters'] = southamerica0['clusters'].map({ 0 : 'Joyful', 1: 'Party', 2 : 'Chill', 3 : 'Romantic'})
southamerica1['clusters'] = southamerica1['clusters'].map({ 0 : 'Joyful', 1: 'Party', 2 : 'Chill', 3 : 'Romantic'})
southamerica2['clusters'] = southamerica2['clusters'].map({ 0 : 'Joyful', 1: 'Party', 2 : 'Chill', 3 : 'Romantic'})
southamerica3['clusters'] = southamerica3['clusters'].map({ 0 : 'Joyful', 1: 'Party', 2 : 'Chill', 3 : 'Romantic'})
southamericasample0 = southamerica0.sample(10)
fig_table_southamerica0 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[southamericasample0.track_name, southamericasample0.artist, southamericasample0.album, southamericasample0.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_southamerica0.update_layout(title_text='Joyful tracks in southamerica')
southamericasample1 = southamerica1.sample(10)
fig_table_southamerica1 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[southamericasample1.track_name, southamericasample1.artist, southamericasample1.album, southamericasample1.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_southamerica1.update_layout(title_text='Party tracks in southamerica')
southamericasample2 = southamerica2.sample(10)
fig_table_southamerica2 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[southamericasample2.track_name, southamericasample2.artist, southamericasample2.album, southamericasample2.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_southamerica2.update_layout(title_text='Chill tracks in southamerica')
southamericasample3 = southamerica3.sample(10)
fig_table_southamerica3 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[southamericasample3.track_name, southamericasample3.artist, southamericasample3.album, southamericasample3.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_southamerica3.update_layout(title_text='Romantic tracks in southamerica')

northamerica0 = df_northamerica_results.loc[df_northamerica_results.loc[:, 'clusters'] == 0]
northamerica1 = df_northamerica_results.loc[df_northamerica_results.loc[:, 'clusters'] == 1]
northamerica2 = df_northamerica_results.loc[df_northamerica_results.loc[:, 'clusters'] == 2]
northamerica3 = df_northamerica_results.loc[df_northamerica_results.loc[:, 'clusters'] == 3]
northamericasample0 = northamerica0.sample(10)
fig_table_northamerica0 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[northamericasample0.track_name, northamericasample0.artist, northamericasample0.album, northamericasample0.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_northamerica0.update_layout(title_text='Cluster 0 tracks in northamerica')
northamericasample1 = northamerica1.sample(10)
fig_table_northamerica1 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[northamericasample1.track_name, northamericasample1.artist, northamericasample1.album, northamericasample1.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_northamerica1.update_layout(title_text='Cluster 1 tracks in northamerica')
northamericasample2 = northamerica2.sample(10)
fig_table_northamerica2 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[northamericasample2.track_name, northamericasample2.artist, northamericasample2.album, northamericasample2.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_northamerica2.update_layout(title_text='Cluster 2 tracks in northamerica')
northamericasample3 = northamerica3.sample(10)
fig_table_northamerica3 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[northamericasample3.track_name, northamericasample3.artist, northamericasample3.album, northamericasample3.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_northamerica3.update_layout(title_text='Cluster 3 tracks in northamerica')



europe0 = df_europe_results.loc[df_europe_results.loc[:, 'clusters'] == 0]
europe1 = df_europe_results.loc[df_europe_results.loc[:, 'clusters'] == 1]
europe2 = df_europe_results.loc[df_europe_results.loc[:, 'clusters'] == 2]
europe3 = df_europe_results.loc[df_europe_results.loc[:, 'clusters'] == 3]
europesample0 = europe0.sample(10)
fig_table_europe0 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[europesample0.track_name, europesample0.artist, europesample0.album, europesample0.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_europe0.update_layout(title_text='Cluster 0 tracks in europe')
europesample1 = europe1.sample(10)
fig_table_europe1 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[europesample1.track_name, europesample1.artist, europesample1.album, europesample1.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_europe1.update_layout(title_text='Cluster 1 tracks in europe')
europesample2 = europe2.sample(10)
fig_table_europe2 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[europesample2.track_name, europesample2.artist, europesample2.album, europesample2.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_europe2.update_layout(title_text='Cluster 2 tracks in europe')
europesample3 = europe3.sample(10)
fig_table_europe3 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[europesample3.track_name, europesample3.artist, europesample3.album, europesample3.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_europe3.update_layout(title_text='Cluster 3 tracks in europe')


asia0 = df_asia_results.loc[df_asia_results.loc[:, 'clusters'] == 0]
asia1 = df_asia_results.loc[df_asia_results.loc[:, 'clusters'] == 1]
asia2 = df_asia_results.loc[df_asia_results.loc[:, 'clusters'] == 2]
asia3 = df_asia_results.loc[df_asia_results.loc[:, 'clusters'] == 3]
asiasample0 = asia0.sample(10)
fig_table_asia0 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[asiasample0.track_name, asiasample0.artist, asiasample0.album, asiasample0.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_asia0.update_layout(title_text='Cluster 0 tracks in asia')
asiasample1 = asia1.sample(10)
fig_table_asia1 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[asiasample1.track_name, asiasample1.artist, asiasample1.album, asiasample1.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_asia1.update_layout(title_text='Cluster 1 tracks in asia')
asiasample2 = asia2.sample(10)
fig_table_asia2 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[asiasample2.track_name, asiasample2.artist, asiasample2.album, asiasample2.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_asia2.update_layout(title_text='Cluster 2 tracks in asia')
asiasample3 = asia3.sample(10)
fig_table_asia3 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[asiasample3.track_name, asiasample3.artist, asiasample3.album, asiasample3.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_asia3.update_layout(title_text='Cluster 3 tracks in asia')

africa0 = df_africa_results.loc[df_africa_results.loc[:, 'clusters'] == 0]
africa1 = df_africa_results.loc[df_africa_results.loc[:, 'clusters'] == 1]
africa2 = df_africa_results.loc[df_africa_results.loc[:, 'clusters'] == 2]
africa3 = df_africa_results.loc[df_africa_results.loc[:, 'clusters'] == 3]
africa0['clusters'] = africa0['clusters'].map({ 0 : 'Joyful', 1: 'Party', 2 : 'Chill', 3 : 'Romantic'})
africa1['clusters'] = africa1['clusters'].map({ 0 : 'Joyful', 1: 'Party', 2 : 'Chill', 3 : 'Romantic'})
africa2['clusters'] = africa2['clusters'].map({ 0 : 'Joyful', 1: 'Party', 2 : 'Chill', 3 : 'Romantic'})
africa3['clusters'] = africa3['clusters'].map({ 0 : 'Joyful', 1: 'Party', 2 : 'Chill', 3 : 'Romantic'})
africasample0 = africa0.sample(10)
fig_table_africa0 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[africasample0.track_name, africasample0.artist, africasample0.album, africasample0.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_africa0.update_layout(title_text='Joyful tracks in africa')
africasample1 = africa1.sample(10)
fig_table_africa1 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[africasample1.track_name, africasample1.artist, africasample1.album, africasample1.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_africa1.update_layout(title_text='Party tracks in africa')
africasample2 = africa2.sample(10)
fig_table_africa2 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[africasample2.track_name, africasample2.artist, africasample2.album, africasample2.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_africa2.update_layout(title_text='Chill tracks in africa')
africasample3 = africa3.sample(10)
fig_table_africa3 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[africasample3.track_name, africasample3.artist, africasample3.album, africasample3.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_africa3.update_layout(title_text='Romantic tracks in africa')


oceania0 = df_oceania_results.loc[df_oceania_results.loc[:, 'clusters'] == 0]
oceania1 = df_oceania_results.loc[df_oceania_results.loc[:, 'clusters'] == 1]
oceania2 = df_oceania_results.loc[df_oceania_results.loc[:, 'clusters'] == 2]
oceania3 = df_oceania_results.loc[df_oceania_results.loc[:, 'clusters'] == 3]
oceaniasample0 = oceania0.sample(10)
fig_table_oceania0 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[oceaniasample0.track_name, oceaniasample0.artist, oceaniasample0.album, oceaniasample0.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_oceania0.update_layout(title_text='Cluster 0 tracks in oceania')
oceaniasample1 = oceania1.sample(10)
fig_table_oceania1 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[oceaniasample1.track_name, oceaniasample1.artist, oceaniasample1.album, oceaniasample1.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_oceania1.update_layout(title_text='Cluster 1 tracks in oceania')
oceaniasample2 = oceania2.sample(10)
fig_table_oceania2 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[oceaniasample2.track_name, oceaniasample2.artist, oceaniasample2.album, oceaniasample2.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_oceania2.update_layout(title_text='Cluster 2 tracks in oceania')
oceaniasample3 = oceania3.sample(10)
fig_table_oceania3 = go.Figure(data=[go.Table(
    header=dict(values=['Track name','Artist','Album','Cluster'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[oceaniasample3.track_name, oceaniasample3.artist, oceaniasample3.album, oceaniasample3.clusters],
               fill_color='lavender',
               align='left'))
])
fig_table_oceania3.update_layout(title_text='Cluster 3 tracks in oceania')


# Histograms
group_labels = ['Cluster 0: Chill','Cluster 1: Romantic','Cluster 2: Party','Cluster 3: Joyful']
colors = ['#656565', '#808782', '#A6D3A0', '#D1FFD7']

x1_energy = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_energy']
x2_energy = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_energy']
x3_energy = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_energy']
x4_energy = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_energy']

x1_acousticness = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_acousticness']
x2_acousticness = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_acousticness']
x3_acousticness = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_acousticness']
x4_acousticness = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_acousticness']

x1_valence = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_valence']
x2_valence = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_valence']
x3_valence = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_valence']
x4_valence = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_valence']

x1_danceability = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_danceability']
x2_danceability = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_danceability']
x3_danceability = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_danceability']
x4_danceability = df_peru_results.loc[df_peru_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_danceability']

fig_histo_peru_energy = ff.create_distplot([x1_energy, x2_energy, x3_energy, x4_energy], group_labels, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_peru_energy.update_layout(title_text='Energy in Peru')

fig_histo_peru_acousticness = ff.create_distplot([x1_acousticness, x2_acousticness, x3_acousticness, x4_acousticness], group_labels, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_peru_acousticness.update_layout(title_text='Acousticness in Peru')

fig_histo_peru_valence = ff.create_distplot([x1_valence, x2_valence, x3_valence, x4_valence], group_labels, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_peru_valence.update_layout(title_text='Valence in Peru')

fig_histo_peru_danceability = ff.create_distplot([x1_danceability, x2_danceability, x3_danceability, x4_danceability], group_labels, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_peru_danceability.update_layout(title_text='Danceability in Peru')

group_labels_southamerica = ['Cluster 0: Joyful','Cluster 1: Party','Cluster 2: Chill','Cluster 3: Romantic']

x1_energy_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_energy']
x2_energy_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_energy']
x3_energy_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_energy']
x4_energy_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_energy']

x1_acousticness_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_acousticness']
x2_acousticness_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_acousticness']
x3_acousticness_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_acousticness']
x4_acousticness_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_acousticness']

x1_valence_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_valence']
x2_valence_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_valence']
x3_valence_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_valence']
x4_valence_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_valence']

x1_danceability_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_danceability']
x2_danceability_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_danceability']
x3_danceability_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_danceability']
x4_danceability_southamerica = df_southamerica_results.loc[df_southamerica_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_danceability']

fig_histo_southamerica_energy = ff.create_distplot([x1_energy_southamerica, x2_energy_southamerica, x3_energy_southamerica, x4_energy_southamerica], group_labels_southamerica, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_southamerica_energy.update_layout(title_text='Energy in Southamerica')

fig_histo_southamerica_acousticness = ff.create_distplot([x1_acousticness_southamerica, x2_acousticness_southamerica, x3_acousticness_southamerica, x4_acousticness_southamerica], group_labels_southamerica, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_southamerica_acousticness.update_layout(title_text='Acousticness in Southamerica')

fig_histo_southamerica_valence = ff.create_distplot([x1_valence_southamerica, x2_valence_southamerica, x3_valence_southamerica, x4_valence_southamerica], group_labels_southamerica, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_southamerica_valence.update_layout(title_text='Valence in Southamerica')

fig_histo_southamerica_danceability = ff.create_distplot([x1_danceability_southamerica, x2_danceability_southamerica, x3_danceability_southamerica, x4_danceability_southamerica], group_labels_southamerica, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_southamerica_danceability.update_layout(title_text='Danceability in Southamerica')


group_labels_unknown = ['Cluster 0','Cluster 1','Cluster 2','Cluster 3']

x1_energy_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_energy']
x2_energy_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_energy']
x3_energy_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_energy']
x4_energy_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_energy']

x1_acousticness_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_acousticness']
x2_acousticness_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_acousticness']
x3_acousticness_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_acousticness']
x4_acousticness_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_acousticness']

x1_valence_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_valence']
x2_valence_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_valence']
x3_valence_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_valence']
x4_valence_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_valence']

x1_danceability_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_danceability']
x2_danceability_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_danceability']
x3_danceability_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_danceability']
x4_danceability_northamerica = df_northamerica_results.loc[df_northamerica_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_danceability']

fig_histo_northamerica_energy = ff.create_distplot([x1_energy_northamerica, x2_energy_northamerica, x3_energy_northamerica, x4_energy_northamerica], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_northamerica_energy.update_layout(title_text='Energy in northamerica')

fig_histo_northamerica_acousticness = ff.create_distplot([x1_acousticness_northamerica, x2_acousticness_northamerica, x3_acousticness_northamerica, x4_acousticness_northamerica], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_northamerica_acousticness.update_layout(title_text='Acousticness in northamerica')

fig_histo_northamerica_valence = ff.create_distplot([x1_valence_northamerica, x2_valence_northamerica, x3_valence_northamerica, x4_valence_northamerica], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_northamerica_valence.update_layout(title_text='Valence in northamerica')

fig_histo_northamerica_danceability = ff.create_distplot([x1_danceability_northamerica, x2_danceability_northamerica, x3_danceability_northamerica, x4_danceability_northamerica], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_northamerica_danceability.update_layout(title_text='Danceability in northamerica')


x1_energy_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_energy']
x2_energy_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_energy']
x3_energy_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_energy']
x4_energy_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_energy']

x1_acousticness_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_acousticness']
x2_acousticness_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_acousticness']
x3_acousticness_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_acousticness']
x4_acousticness_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_acousticness']

x1_valence_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_valence']
x2_valence_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_valence']
x3_valence_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_valence']
x4_valence_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_valence']

x1_danceability_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_danceability']
x2_danceability_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_danceability']
x3_danceability_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_danceability']
x4_danceability_europe = df_europe_results.loc[df_europe_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_danceability']

fig_histo_europe_energy = ff.create_distplot([x1_energy_europe, x2_energy_europe, x3_energy_europe, x4_energy_europe], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_europe_energy.update_layout(title_text='Energy in europe')

fig_histo_europe_acousticness = ff.create_distplot([x1_acousticness_europe, x2_acousticness_europe, x3_acousticness_europe, x4_acousticness_europe], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_europe_acousticness.update_layout(title_text='Acousticness in europe')

fig_histo_europe_valence = ff.create_distplot([x1_valence_europe, x2_valence_europe, x3_valence_europe, x4_valence_europe], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_europe_valence.update_layout(title_text='Valence in europe')

fig_histo_europe_danceability = ff.create_distplot([x1_danceability_europe, x2_danceability_europe, x3_danceability_europe, x4_danceability_europe], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_europe_danceability.update_layout(title_text='Danceability in europe')



x1_energy_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_energy']
x2_energy_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_energy']
x3_energy_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_energy']
x4_energy_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_energy']

x1_acousticness_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_acousticness']
x2_acousticness_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_acousticness']
x3_acousticness_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_acousticness']
x4_acousticness_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_acousticness']

x1_valence_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_valence']
x2_valence_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_valence']
x3_valence_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_valence']
x4_valence_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_valence']

x1_danceability_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_danceability']
x2_danceability_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_danceability']
x3_danceability_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_danceability']
x4_danceability_asia = df_asia_results.loc[df_asia_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_danceability']

fig_histo_asia_energy = ff.create_distplot([x1_energy_asia, x2_energy_asia, x3_energy_asia, x4_energy_asia], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_asia_energy.update_layout(title_text='Energy in asia')

fig_histo_asia_acousticness = ff.create_distplot([x1_acousticness_asia, x2_acousticness_asia, x3_acousticness_asia, x4_acousticness_asia], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_asia_acousticness.update_layout(title_text='Acousticness in asia')

fig_histo_asia_valence = ff.create_distplot([x1_valence_asia, x2_valence_asia, x3_valence_asia, x4_valence_asia], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_asia_valence.update_layout(title_text='Valence in asia')

fig_histo_asia_danceability = ff.create_distplot([x1_danceability_asia, x2_danceability_asia, x3_danceability_asia, x4_danceability_asia], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_asia_danceability.update_layout(title_text='Danceability in asia')



x1_energy_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_energy']
x2_energy_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_energy']
x3_energy_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_energy']
x4_energy_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_energy']

x1_acousticness_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_acousticness']
x2_acousticness_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_acousticness']
x3_acousticness_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_acousticness']
x4_acousticness_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_acousticness']

x1_valence_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_valence']
x2_valence_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_valence']
x3_valence_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_valence']
x4_valence_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_valence']

x1_danceability_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_danceability']
x2_danceability_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_danceability']
x3_danceability_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_danceability']
x4_danceability_africa = df_africa_results.loc[df_africa_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_danceability']

fig_histo_africa_energy = ff.create_distplot([x1_energy_africa, x2_energy_africa, x3_energy_africa, x4_energy_africa], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_africa_energy.update_layout(title_text='Energy in africa')

fig_histo_africa_acousticness = ff.create_distplot([x1_acousticness_africa, x2_acousticness_africa, x3_acousticness_africa, x4_acousticness_africa], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_africa_acousticness.update_layout(title_text='Acousticness in africa')

fig_histo_africa_valence = ff.create_distplot([x1_valence_africa, x2_valence_africa, x3_valence_africa, x4_valence_africa], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_africa_valence.update_layout(title_text='Valence in africa')

fig_histo_africa_danceability = ff.create_distplot([x1_danceability_africa, x2_danceability_africa, x3_danceability_africa, x4_danceability_africa], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_africa_danceability.update_layout(title_text='Danceability in africa')



x1_energy_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_energy']
x2_energy_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_energy']
x3_energy_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_energy']
x4_energy_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_energy']

x1_acousticness_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_acousticness']
x2_acousticness_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_acousticness']
x3_acousticness_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_acousticness']
x4_acousticness_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_acousticness']

x1_valence_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_valence']
x2_valence_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_valence']
x3_valence_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_valence']
x4_valence_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_valence']

x1_danceability_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 0].loc[:, 'track_danceability']
x2_danceability_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 1].loc[:, 'track_danceability']
x3_danceability_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 2].loc[:, 'track_danceability']
x4_danceability_oceania = df_oceania_results.loc[df_oceania_results.\
     loc[:, 'clusters'] == 3].loc[:, 'track_danceability']

fig_histo_oceania_energy = ff.create_distplot([x1_energy_oceania, x2_energy_oceania, x3_energy_oceania, x4_energy_oceania], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_oceania_energy.update_layout(title_text='Energy in oceania')

fig_histo_oceania_acousticness = ff.create_distplot([x1_acousticness_oceania, x2_acousticness_oceania, x3_acousticness_oceania, x4_acousticness_oceania], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_oceania_acousticness.update_layout(title_text='Acousticness in oceania')

fig_histo_oceania_valence = ff.create_distplot([x1_valence_oceania, x2_valence_oceania, x3_valence_oceania, x4_valence_oceania], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_oceania_valence.update_layout(title_text='Valence in oceania')

fig_histo_oceania_danceability = ff.create_distplot([x1_danceability_oceania, x2_danceability_oceania, x3_danceability_oceania, x4_danceability_oceania], group_labels_unknown, colors=colors,
                         bin_size=[0.05, 0.05, 0.05, 0.05], show_hist=True, show_rug=False)
fig_histo_oceania_danceability.update_layout(title_text='Danceability in oceania')


# Stacked bar
df_stacked_bar_peru = df_peru.join(df_peru_results.set_index('track_id'), on='track_id', rsuffix='_df_peru').loc[:, ['date_extraction', 'track_id', 'clusters']]
df_stacked_bar_peru.loc[:, 'date_extraction'] = df_stacked_bar_peru.loc[:, 'date_extraction'].astype('datetime64[ns]')
df_stacked_bar_peru.reset_index(inplace=True, drop=True)
df_stacked_bar_peru.drop(index=20560, inplace=True)
df_stacked_bar_peru.reset_index(inplace=True, drop=True)
df_stacked_bar_peru.loc[:, 'clusters'] = df_stacked_bar_peru.loc[:, 'clusters'].astype(int)
df_stacked_bar_peru = df_stacked_bar_peru.groupby(by=['date_extraction', 'clusters'], as_index=False).count()

fig_stacked_peru = go.Figure()
fig_stacked_peru.add_trace(go.Scatter(
    x=df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 0].loc[:, 'date_extraction'], y=df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 0].loc[:, 'track_id'],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='#AFC2D5'),
    stackgroup='one', # define stack group
    name='Chill'
))
fig_stacked_peru.add_trace(go.Scatter(
    x=df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 1].loc[:, 'date_extraction'], y=df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 1].loc[:, 'track_id'],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='#CCDDD3'),
    stackgroup='one',
    name='Romantic'
))
fig_stacked_peru.add_trace(go.Scatter(
    x=df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 2].loc[:, 'date_extraction'], y=df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 2].loc[:, 'track_id'],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='#FFF9A5'),
    stackgroup='one',
    name='Party'
))
fig_stacked_peru.add_trace(go.Scatter(
    x=df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 3].loc[:, 'date_extraction'], y=df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 3].loc[:, 'track_id'],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='#B48B7D'),
    stackgroup='one',
    name='Joyful'
))

fig_stacked_peru0 = px.line(df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 0], x='date_extraction', y='track_id', title='Chill Tracks')
fig_stacked_peru0.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig_stacked_peru1 = px.line(df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 1], x='date_extraction', y='track_id', title='Romantic Tracks')
fig_stacked_peru1.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig_stacked_peru2 = px.line(df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 2], x='date_extraction', y='track_id', title='Party tracks')
fig_stacked_peru2.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig_stacked_peru3 = px.line(df_stacked_bar_peru.loc[df_stacked_bar_peru.loc[:, 'clusters'] == 3], x='date_extraction', y='track_id', title='Joyful tracks')
fig_stacked_peru3.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

df_stacked_bar_southamerica = df_southamerica.join(df_southamerica_results.set_index('track_id'), on='track_id', rsuffix='_df_southamerica').loc[:, ['date_extraction', 'track_id', 'clusters']]
df_stacked_bar_southamerica.loc[:, 'date_extraction'] = df_stacked_bar_southamerica.loc[:, 'date_extraction'].astype('datetime64[ns]')
df_stacked_bar_southamerica.reset_index(inplace=True, drop=True)
df_stacked_bar_southamerica.loc[:, 'clusters'] = df_stacked_bar_southamerica.loc[:, 'clusters'].astype(int)
df_stacked_bar_southamerica = df_stacked_bar_southamerica.groupby(by=['date_extraction', 'clusters'], as_index=False).count()

fig_stacked_southamerica = go.Figure()
fig_stacked_southamerica.add_trace(go.Scatter(
    x=df_stacked_bar_southamerica.loc[df_stacked_bar_southamerica.loc[:, 'clusters'] == 0].loc[:, 'date_extraction'], y=df_stacked_bar_southamerica.loc[df_stacked_bar_southamerica.loc[:, 'clusters'] == 0].loc[:, 'track_id'],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='#AFC2D5'),
    stackgroup='one', # define stack group
    name='Joyful'
))
fig_stacked_southamerica.add_trace(go.Scatter(
    x=df_stacked_bar_southamerica.loc[df_stacked_bar_southamerica.loc[:, 'clusters'] == 1].loc[:, 'date_extraction'], y=df_stacked_bar_southamerica.loc[df_stacked_bar_southamerica.loc[:, 'clusters'] == 1].loc[:, 'track_id'],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='#CCDDD3'),
    stackgroup='one',
    name='Party'
))
fig_stacked_southamerica.add_trace(go.Scatter(
    x=df_stacked_bar_southamerica.loc[df_stacked_bar_southamerica.loc[:, 'clusters'] == 2].loc[:, 'date_extraction'], y=df_stacked_bar_southamerica.loc[df_stacked_bar_southamerica.loc[:, 'clusters'] == 2].loc[:, 'track_id'],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='#FFF9A5'),
    stackgroup='one',
    name='Chill'
))
fig_stacked_southamerica.add_trace(go.Scatter(
    x=df_stacked_bar_southamerica.loc[df_stacked_bar_southamerica.loc[:, 'clusters'] == 3].loc[:, 'date_extraction'], y=df_stacked_bar_southamerica.loc[df_stacked_bar_southamerica.loc[:, 'clusters'] == 3].loc[:, 'track_id'],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='#B48B7D'),
    stackgroup='one',
    name='Romantic'
))


app.layout = html.Div([
    html.H1(children='Spotify Tracks Analysis in Lockdown', style={'textAlign': 'center'}),
    html.Div(children='Students: Ronie Arauco & Handry Guillen', style={'textAlign': 'center'}),
    html.Div(),
    dcc.Tabs([
        dcc.Tab(label='Peru Tracks', children=[
            dcc.Graph(
                id='valence-peru',
                figure=fig_valence_peru
            ),
            dcc.Graph(
                id='pie-chart-peru',
                figure=fig_pie_peru
            ),
            dcc.Graph(
                id='table-peru0',
                figure=fig_table_peru0
            ),
            dcc.Graph(
                id='table-peru1',
                figure=fig_table_peru1
            ),
            dcc.Graph(
                id='table-peru2',
                figure=fig_table_peru2
            ),
            dcc.Graph(
                id='table-peru3',
                figure=fig_table_peru3
            ),
            dcc.Graph(
                id='histo-peru-energy',
                figure=fig_histo_peru_energy
            ),
            dcc.Graph(
                id='histo-peru-acousticness',
                figure=fig_histo_peru_acousticness
            ),
            dcc.Graph(
                id='histo-peru-valence',
                figure=fig_histo_peru_valence
            ),
            dcc.Graph(
                id='histo-peru-danceability',
                figure=fig_histo_peru_danceability
            ),
            dcc.Graph(
                id='stacked-peru',
                figure=fig_stacked_peru
            ),
            dcc.Graph(
                id='stacked-peru0',
                figure=fig_stacked_peru0
            ),
            dcc.Graph(
                id='stacked-peru1',
                figure=fig_stacked_peru1
            ),
            dcc.Graph(
                id='stacked-peru2',
                figure=fig_stacked_peru2
            ),
            dcc.Graph(
                id='stacked-peru3',
                figure=fig_stacked_peru3
            )
        ]),
        dcc.Tab(label='Southamerica Tracks', children=[
            dcc.Graph(
                id='valence-southamerica',
                figure=fig_valence_southamerica
            ),
            dcc.Graph(
                id='pie-chart-southamerica',
                figure=fig_pie_southamerica
            ),
            dcc.Graph(
                id='table-southamerica0',
                figure=fig_table_southamerica0
            ),
            dcc.Graph(
                id='table-southamerica1',
                figure=fig_table_southamerica1
            ),
            dcc.Graph(
                id='table-southamerica2',
                figure=fig_table_southamerica2
            ),
            dcc.Graph(
                id='table-southamerica3',
                figure=fig_table_southamerica3
            ),
            dcc.Graph(
                id='histo-southamerica-energy',
                figure=fig_histo_southamerica_energy
            ),
            dcc.Graph(
                id='histo-southamerica-acousticness',
                figure=fig_histo_southamerica_acousticness
            ),
            dcc.Graph(
                id='histo-southamerica-valence',
                figure=fig_histo_southamerica_valence
            ),
            dcc.Graph(
                id='histo-southamerica-danceability',
                figure=fig_histo_southamerica_danceability
            )
        ]),
        dcc.Tab(label='Northamerica Tracks', children=[
            dcc.Graph(
                id='valence-northamerica',
                figure=fig_valence_northamerica
            ),
            dcc.Graph(
                id='pie-chart-northamerica',
                figure=fig_pie_northamerica
            ),
            dcc.Graph(
                id='table-northamerica0',
                figure=fig_table_northamerica0
            ),
            dcc.Graph(
                id='table-northamerica1',
                figure=fig_table_northamerica1
            ),
            dcc.Graph(
                id='table-northamerica2',
                figure=fig_table_northamerica2
            ),
            dcc.Graph(
                id='table-northamerica3',
                figure=fig_table_northamerica3
            ),
            dcc.Graph(
                id='histo-northamerica-energy',
                figure=fig_histo_northamerica_energy
            ),
            dcc.Graph(
                id='histo-northamerica-acousticness',
                figure=fig_histo_northamerica_acousticness
            ),
            dcc.Graph(
                id='histo-northamerica-valence',
                figure=fig_histo_northamerica_valence
            ),
            dcc.Graph(
                id='histo-northamerica-danceability',
                figure=fig_histo_northamerica_danceability
            )
        ]),
        dcc.Tab(label='Europe Tracks', children=[
            dcc.Graph(
                id='valence-europe',
                figure=fig_valence_europe
            ),
            dcc.Graph(
                id='pie-chart-europe',
                figure=fig_pie_europe
            ),
            dcc.Graph(
                id='table-europe0',
                figure=fig_table_europe0
            ),
            dcc.Graph(
                id='table-europe',
                figure=fig_table_europe1
            ),
            dcc.Graph(
                id='table-europe2',
                figure=fig_table_europe2
            ),
            dcc.Graph(
                id='table-europe3',
                figure=fig_table_europe3
            ),
            dcc.Graph(
                id='histo-europe-energy',
                figure=fig_histo_europe_energy
            ),
            dcc.Graph(
                id='histo-europe-acousticness',
                figure=fig_histo_europe_acousticness
            ),
            dcc.Graph(
                id='histo-europe-valence',
                figure=fig_histo_europe_valence
            ),
            dcc.Graph(
                id='histo-europe-danceability',
                figure=fig_histo_europe_danceability
            )
        ]),
        dcc.Tab(label='Asia Tracks', children=[
            dcc.Graph(
                id='valence-asia',
                figure=fig_valence_asia
            ),
            dcc.Graph(
                id='pie-chart-asia',
                figure=fig_pie_asia
            ),
            dcc.Graph(
                id='table-asia0',
                figure=fig_table_asia0
            ),
            dcc.Graph(
                id='table-asia1',
                figure=fig_table_asia1
            ),
            dcc.Graph(
                id='table-asia2',
                figure=fig_table_asia2
            ),
            dcc.Graph(
                id='table-asia3',
                figure=fig_table_asia3
            ),
            dcc.Graph(
                id='histo-asia-energy',
                figure=fig_histo_asia_energy
            ),
            dcc.Graph(
                id='histo-asia-acousticness',
                figure=fig_histo_asia_acousticness
            ),
            dcc.Graph(
                id='histo-asia-valence',
                figure=fig_histo_asia_valence
            ),
            dcc.Graph(
                id='histo-asia-danceability',
                figure=fig_histo_asia_danceability
            )
        ]),
        dcc.Tab(label='Southafrica Tracks', children=[
            dcc.Graph(
                id='valence-africa',
                figure=fig_valence_africa
            ),
            dcc.Graph(
                id='pie-chart-africa',
                figure=fig_pie_africa
            ),
            dcc.Graph(
                id='table-africa0',
                figure=fig_table_africa0
            ),
            dcc.Graph(
                id='table-africa1',
                figure=fig_table_africa1
            ),
            dcc.Graph(
                id='table-africa2',
                figure=fig_table_africa2
            ),
            dcc.Graph(
                id='table-africa3',
                figure=fig_table_africa3
            ),
            dcc.Graph(
                id='histo-africa-energy',
                figure=fig_histo_africa_energy
            ),
            dcc.Graph(
                id='histo-africa-acousticness',
                figure=fig_histo_africa_acousticness
            ),
            dcc.Graph(
                id='histo-africa-valence',
                figure=fig_histo_africa_valence
            ),
            dcc.Graph(
                id='histo-africa-danceability',
                figure=fig_histo_africa_danceability
            )
        ]),
        dcc.Tab(label='Oceania Tracks', children=[
            dcc.Graph(
                id='valence-oceania',
                figure=fig_valence_oceania
            ),
            dcc.Graph(
                id='pie-chart-oceania',
                figure=fig_pie_oceania
            ),
            dcc.Graph(
                id='table-oceania0',
                figure=fig_table_oceania0
            ),
            dcc.Graph(
                id='table-oceania1',
                figure=fig_table_oceania1
            ),
            dcc.Graph(
                id='table-oceania2',
                figure=fig_table_oceania2
            ),
            dcc.Graph(
                id='table-oceania3',
                figure=fig_table_oceania3
            ),
            dcc.Graph(
                id='histo-oceania-energy',
                figure=fig_histo_oceania_energy
            ),
            dcc.Graph(
                id='histo-oceania-acousticness',
                figure=fig_histo_oceania_acousticness
            ),
            dcc.Graph(
                id='histo-oceania-valence',
                figure=fig_histo_oceania_valence
            ),
            dcc.Graph(
                id='histo-oceania-danceability',
                figure=fig_histo_oceania_danceability
            )
        ]),
        dcc.Tab(label='About', children=[
            html.Div(html.Img(src='https://dl-ta-dataset.s3.amazonaws.com/expo2.png'), style={'textAlign': 'center'})
        ]),
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True)
