# -*- coding: utf-8 -*-
import time
import os
import json
import tarfile
import shutil
import requests
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf

import dash
import dash_core_components as dcc
import dash_dangerously_set_inner_html
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq
from plotly.colors import sequential
from pyproj import Transformer
import dask_cudf
from dask import delayed
from distributed import Client
from dask_cuda import LocalCUDACluster
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime
import cudf
import pandas as pd
import cupy

# Disable cupy memory pool so that cupy immediately releases GPU memory
cupy.cuda.set_allocator(None)

# Colors
bgcolor = "#191a1a"  # mapbox dark map land color
text_color = "#cfd8dc"  # Material blue-grey 100
mapbox_land_color = "#343332"
covid_last_update_time = time.time()
covid_last_update_date = datetime.datetime.today().strftime("%b-%d-%Y at %H:%M")

# Figure template
row_heights = [200, 460, 200, 1600, 300, 100]
template = {
    'layout': {
        'paper_bgcolor': bgcolor,
        'plot_bgcolor': bgcolor,
        'font': {'color': text_color},
        "margin": {"r": 0, "t": 30, "l": 0, "b": 20},
        'bargap': 0.05,
        'xaxis': {'showgrid': False, 'automargin': True},
        'yaxis': {'showgrid': True, 'automargin': True, 'gridcolor': 'rgba(139, 139, 139, 0.1)'},
    }
}

colors = {
    'age': ['#C700E5','#9D00DB', '#7300D2','#4900C8','#1F00BF'],
    # 'age': ['red','green', 'blue','orange']
}
mappings = {}

# Load mapbox token from environment variable or file
token = os.getenv('MAPBOX_TOKEN')
if not token:
    token = open(".mapbox_token").read()


# Names of float columns
float_columns = [
    'x', 'y'
]
cache_fig_1, cache_fig_2, cache_fig_3, cache_fig_4 = None, None, None, None
data_center_3857, data_3857, data_4326, data_center_4326 = [], [], [], []

def load_dataset(path):
    """
    Args:
        path: Path to arrow file containing census2010 dataset

    Returns:
        cudf DataFrame
    """
    if os.path.isdir(path):
        path = path + '/*'
    df_d = cudf.read_parquet(path)
    if 'age' in df_d.columns:
        df_d.drop_column('sex')
        # df_d.drop_column('age')
    # df_d = df_d.head(220_000_000)
        df_d['age'][df_d['age']<=20] = 1
        mask_ = (df_d['age']>20) & (df_d['age']<=40)
        df_d['age'][mask_] = 2
        mask_ = (df_d['age']>40) & (df_d['age']<=50)
        df_d['age'][mask_] = 3
        mask_ = (df_d['age']>50) & (df_d['age']<=60)
        df_d['age'][mask_] = 4
        del(mask_)
        df_d['age'][df_d['age']>60] = 5
        df_d['age'] = df_d.age.to_pandas().astype('category')
    return df_d


def resolve_missing_counties(df, df_):
    '''
        making sure counties reported yesterday and today are consistent in size
    '''
    test_df = df.merge(df_, on=['COUNTY'], how='outer', suffixes=['', '_'])
    df_not_in_today = test_df[test_df.Deaths.isna()]
    for i in df.columns:
        if i != 'COUNTY':
            df_not_in_today[i] = df_not_in_today[str(i)+'_']
    df_not_in_today = df_not_in_today[df.columns]
    df = cudf.concat([df, df_not_in_today]).sort_values('COUNTY').reset_index()
    
    df_not_in_yesterday = test_df[test_df.Deaths_.isna()]
    for i in df_.columns:
        if i == 'Confirmed' or i == 'Deaths':
            df_not_in_yesterday[str(i)] = 0
    df_not_in_yesterday['Last_Update'] = df_not_in_yesterday['Last_Update_']
    df_not_in_yesterday = df_not_in_yesterday[df_.columns]
    df_ = cudf.concat([df_, df_not_in_yesterday]).sort_values('COUNTY').reset_index()
    return df, df_

def load_covid(BASE_URL):
    print('loading latest covid dataset...')
    df_temp = []

    last_n_days = (datetime.date.today() - datetime.date(2020, 3, 25)).days
    today = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%m-%d-%Y")

    if requests.get(BASE_URL % (datetime.date.today()).strftime("%m-%d-%Y")).status_code == 200:
        today = (datetime.date.today()).strftime("%m-%d-%Y")

    path = '../data/'+today+'.parquet'
    if not os.path.exists(path):
        print('downloading latest covid dataset...')
        for i in range(last_n_days):
            date_ = str((datetime.date.today() - datetime.timedelta(days=i)).strftime("%m-%d-%Y"))
            if requests.get(BASE_URL % date_).status_code == 404:
                continue
            path = '../data/'+date_+'.parquet'
            if not os.path.exists(path):
                df_temp.append(
                    cudf.from_pandas(
                        pd.read_csv(BASE_URL % date_,
                            usecols=['Lat', 'Long_','Province_State', 'Last_Update', 'Confirmed', 'Deaths', 'Country_Region', 'Combined_Key']
                        ).query('Country_Region == "US" and Confirmed != 0')
                    )
                )
            else:
                if os.path.isdir(path):
                    path = path + '/*'
                df_temp.append(
                    cudf.read_parquet(path)
                )
                break
        df = cudf.concat(df_temp)
        df.to_parquet('../data/'+today+'.parquet')
    else:
        if os.path.isdir(path):
            path = path + '/*'
        print('loading cached latest covid dataset...')
        df = cudf.read_parquet(path)

    df_combined_key = df[['Combined_Key', 'Lat', 'Long_']].dropna()
    df_combined_key.rename({
        'Combined_Key': 'COUNTY'
    }, inplace=True)
    df_combined_key.drop_duplicates(inplace=True)

    df.Last_Update = pd.to_datetime(df.Last_Update.str.split(' ')[0].to_pandas().astype('str'))
    df.rename({
        'Combined_Key': 'COUNTY'
    }, inplace=True)
    df_states_last_n_days = df.groupby(['Province_State','Last_Update']).sum().reset_index().to_pandas()
    df_county = df.groupby(['COUNTY','Last_Update']).agg(
        {
            'Deaths': 'sum', 
            'Confirmed': 'sum',
        }
    ).reset_index()
    df_county = df_county.merge(df_combined_key, on='COUNTY')
    df_county.Last_Update = pd.to_datetime(df_county.Last_Update.str.split(' ')[0].to_pandas().astype('str'))
    last_2_days = np.sort(df_county.Last_Update.unique().to_array())[-2:]
    df_count_latest = df_county.query('Last_Update == @last_2_days[-1]').drop_duplicates('COUNTY').reset_index(drop=True)
    df_count_latest_minus_1 = df_county.query('Last_Update == @last_2_days[0]').drop_duplicates('COUNTY').reset_index(drop=True)
    df_count_latest, df_count_latest_minus_1 = resolve_missing_counties(df_count_latest, df_count_latest_minus_1)
    return (df_states_last_n_days, df_count_latest, df_count_latest_minus_1)

def load_hospitals(path):
    df_hospitals = pd.read_csv(path, usecols=['X', 'Y', 'BEDS', 'NAME'])
    df_hospitals['BEDS_label'] = df_hospitals.BEDS.replace(-999, 'unknown')
    df_hospitals.BEDS = df_hospitals.BEDS.replace(-999, 0)
    df_hospitals['BEDS_sizes'] = df_hospitals.BEDS.apply(lambda x: x if x > 150 else 150)
    return df_hospitals

def set_projection_bounds(df_d):
    transformer_4326_to_3857 = Transformer.from_crs("epsg:4326", "epsg:3857")
    def epsg_4326_to_3857(coords):
        return [transformer_4326_to_3857.transform(*reversed(row)) for row in coords]

    transformer_3857_to_4326 = Transformer.from_crs("epsg:3857", "epsg:4326")
    def epsg_3857_to_4326(coords):
        return [list(reversed(transformer_3857_to_4326.transform(*row))) for row in coords]

    data_3857 = (
        [df_d.x.min(), df_d.y.min()],
        [df_d.x.max(), df_d.y.max()]
    )
    data_center_3857 = [[
        (data_3857[0][0] + data_3857[1][0]) / 2.0,
        (data_3857[0][1] + data_3857[1][1]) / 2.0,
    ]]

    data_4326 = epsg_3857_to_4326(data_3857)
    data_center_4326 = epsg_3857_to_4326(data_center_3857)

    return data_3857, data_center_3857, data_4326, data_center_4326

# Build Dash app and initial layout
def blank_fig(height):
    """
    Build blank figure with the requested height
    Args:
        height: height of blank figure in pixels
    Returns:
        Figure dict
    """
    return {
        'data': [],
        'layout': {
            'height': height,
            'template': template,
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
        }
    }


app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.Div([
        html.H1(children=[
            'US Population 2010 | COVID-19',
            html.A(
                html.Img(
                    src="https://camo.githubusercontent.com/38ca5c5f7d6afc09f8d50fd88abd4b212f0a6375/68747470733a2f2f7261706964732e61692f6173736574732f696d616765732f7261706964735f6c6f676f2e706e67",
                    style={'float': 'right', 'height': '50px', 'margin-right': '2%'}
                ), href="https://rapids.ai/"),
            html.A(
                html.Img(
                    src="assets/dash-logo.png",
                    style={'float': 'right', 'height': '50px', 'margin-right': '2%'}
                ), href="https://dash.plot.ly/"),

        ], style={'text-align': 'left'}),
    ]),
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.Div([
                    dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                        """
                        <h4 class='container_title'> Population (2010)<sup>1</sup> | Known Hospital Beds (2019)<sup>2</sup> | Covid Cases (Daily)<sup>3</sup></h4>
                        """
                    ),
                ], className="container_title"),

                dcc.Loading(
                    dcc.Graph(
                        id='indicator-graph',
                        figure=blank_fig(row_heights[5]),
                        config={'displayModeBar': False},
                    ),
                    style={'height': row_heights[5]},
                )
            ],  style={'height': row_heights[0]}, className='seven columns pretty_container', id="indicator-div"),
            html.Div(children=[
                html.H4([
                    "Display Options",
                ], className="container_title"),
                html.Table([
                    html.Col(style={'width': '100px'}),
                    html.Col(),
                    html.Col(),
                    html.Tr([
                            html.Td(
                                html.Div("Population"), className="config-label"
                            ),
                            html.Td(daq.DarkThemeProvider(daq.BooleanSwitch(
                                on=True,
                                color='#00cc96',
                                id='population-toggle',
                            ))),
                            html.Td(
                                html.Div("Total by"), className="config-label"
                            ),
                            html.Td(dcc.Dropdown(
                                id='colorscale-dropdown-population',
                                options=[
                                    {'label': 'Age by '+cs if cs == 'Purblue' else 'Total by '+cs, 'value': cs}
                                    for cs in ['Purblue', 'Viridis', 'Cividis', 'Inferno', 'Magma', 'Plasma']
                                ],
                                value='Purblue',
                                searchable=False,
                                clearable=False,
                            ), style={'width': '50%', 'height':'15px'}),
                        ]),
                    html.Tr([
                            html.Td(
                                html.Div("Hospitals"), className="config-label"
                            ),
                            html.Td(daq.DarkThemeProvider(daq.BooleanSwitch(
                                on=False,
                                color='#00cc96',
                                id='hospital-toggle',
                            ))),
                            html.Td(
                                html.Div("Color Setting"), className="config-label"
                            ),
                            html.Td(dcc.Dropdown(
                                id='colorscale-dropdown-hospital',
                                options=[
                                    {'label': 'white', 'value': 'white'},
                                    {'label': 'blue', 'value': '#00d0ff'},
                                    {'label': 'orange', 'value': '#ffa41c'}
                                ],
                                value='white',
                                searchable=False,
                                clearable=False,
                            ), style={'width': '50%','height':'15px'}),
                        ]),
                    html.Tr([
                            html.Td(
                                html.Div("COVID-19"), className="config-label"
                            ),
                            html.Td(daq.DarkThemeProvider(daq.BooleanSwitch(
                                on=True,
                                color='#00cc96',
                                id='covid-toggle',
                            ))),
                            html.Td(
                                html.Div("Count Type"), className="config-label"
                            ),
                            html.Td(dcc.Dropdown(
                                id='covid_count_type',
                                options=[
                                    {'label': 'Total Cases', 'value': 0},
                                    {'label': '% increase since last 2 days', 'value': 1},
                                    {'label': 'Case / County Population (2018 ACS)', 'value': 2},
                                ],
                                value=0,
                                searchable=False,
                                clearable=False,
                            ), style={'width': '50%', 'height':'15px'}),
                        ])
                ], style={'width': '100%'}),
            ],   style={'height': row_heights[0]}, className='five columns pretty_container', id="config-div"),
        ]),
        html.Div(children=[
            html.Button("Reset View", id='reset-map', className='reset-button'),
            html.Div([
                dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                        """
                        <h4 class='container_title'> Population Density (2010)<sup>1</sup> | Known Hospital and Beds (2019)<sup>2</sup> | Reported COVID Cases by County (Daily)<sup>3</sup> | Zoom to filter</h4>
                        """
                    )
            ], className="container_title"),
            dcc.Graph(
                id='map-graph',
                config={'displayModeBar': False},
                figure=blank_fig(row_heights[1]),
            ),
        ], className='twelve columns pretty_container',
            style={
                'width': '98%',
                'margin-right': '0',
            },
            id="map-div"
        ),
        html.Div(children=[
            html.Div(
                children=[
                    html.Div(dcc.Dropdown(
                                id='scale-covid-dropdown',
                                options=[
                                    {'label': 'Log scale', 'value': 'log'},
                                    {'label': 'Linear scale', 'value': 'linear'},
                                ],
                                value='linear',
                                searchable=False,
                                clearable=False,
                            ), style={'width': '10%', 'height':'30px', 'float': 'right', 'margin': '0px 10px'}),
                    html.Div(dcc.Dropdown(
                                id='category-covid-dropdown',
                                options=[
                                    {'label': cs, 'value': cs}
                                    for cs in ['Total cases', 'Total deaths']
                                ],
                                value='Total cases',
                                searchable=False,
                                clearable=False,
                            ), style={'width': '10%', 'height':'30px', 'float': 'right'}),
                    html.Div([
                        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                            """
                                <h4 class='container_title'> Reported COVID Cases (total) and Deaths (total) by State <sup>3</sup> </h4>
                            """
                        )
                    ], className="container_title"),

                    dcc.Loading(dcc.Graph(
                        id='covid-histogram',
                        config={'displayModeBar': False},
                        figure=blank_fig(row_heights[3]),
                        animate=True
                    ),
                    style={'height': row_heights[3]}),
                ],
                className='twelve columns pretty_container', id="covid-div"
            )
        ])
    ]),
    html.Div(
        [
            html.H4('Acknowledgments and Data Sources', style={"margin-top": "0"}),
            dcc.Markdown(f'''
- [1] 2010 Population Census and [4] 2018 ACS data used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](www.nhgis.org) ( not for redistribution )
- [2] Hospital data is from [HIFLD](https://hifld-geoplatform.opendata.arcgis.com/datasets/hospitals) (10/7/2019) and does not contain emergency field hospitals
- [3] COVID-19 data is from the [Johns Hopkins University](https://coronavirus.jhu.edu/) data on [GitHub](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports) (last updated {covid_last_update_date})
- Base map layer provided by [mapbox](https://www.mapbox.com/)
- Dashboard developed with Plot.ly [Dash](https://dash.plotly.com/)
- Geospatial point rendering developed with [Datashader](https://datashader.org/)
- GPU accelerated with [RAPIDS](https://rapids.ai/) [cudf](https://github.com/rapidsai/cudf) and [cupy](https://cupy.chainer.org/) libraries
- For more information reach out with this [Covid-19 Slack Channel](https://join.slack.com/t/rapids-goai/shared_invite/zt-2qmkjvzl-K3rVHb1rZYuFeczoR9e4EA)
- For source code visit our [GitHub](https://github.com/rapidsai/plotly-dash-rapids-census-demo)
'''),
        ],
        style={
            'width': '98%',
            'margin-right': '0',
            'padding': '10px',
        },
        className='twelve columns pretty_container',
    ),
])

# Query string helpers
def bar_selection_to_query(selection, column):
    """
    Compute pandas query expression string for selection callback data

    Args:
        selection: selectedData dictionary from Dash callback on a bar trace
        column: Name of the column that the selected bar chart is based on

    Returns:
        String containing a query expression compatible with DataFrame.query. This
        expression will filter the input DataFrame to contain only those rows that
        are contained in the selection.
    """
    point_inds = [p['label'] for p in selection['points']]
    xmin = min(point_inds) #bin_edges[min(point_inds)]
    xmax = max(point_inds) + 1 #bin_edges[max(point_inds) + 1]
    xmin_op = "<="
    xmax_op = "<="
    return f"{xmin} {xmin_op} {column} and {column} {xmax_op} {xmax}"


def build_query(selections, exclude=None):
    """
    Build pandas query expression string for cross-filtered plot

    Args:
        selections: Dictionary from column name to query expression
        exclude: If specified, column to exclude from combined expression

    Returns:
        String containing a query expression compatible with DataFrame.query.
    """
    other_selected = {sel for c, sel in selections.items() if (c != exclude and sel != -1)}
    if other_selected:
        return ' and '.join(other_selected)
    else:
        return None


# Plot functions
def build_colorscale(colorscale_name, transform, agg, agg_col):
    """
    Build plotly colorscale

    Args:
        colorscale_name: Name of a colorscale from the plotly.colors.sequential module
        transform: Transform to apply to colors scale. One of 'linear', 'sqrt', 'cbrt',
        or 'log'

    Returns:
        Plotly color scale list
    """
    global colors, mappings


    colors_temp = getattr(sequential, colorscale_name)
    if transform == "linear":
        scale_values = np.linspace(0, 1, len(colors_temp))
    elif transform == "sqrt":
        scale_values = np.linspace(0, 1, len(colors_temp)) ** 2
    elif transform == "cbrt":
        scale_values = np.linspace(0, 1, len(colors_temp)) ** 3
    elif transform == "log":
        scale_values = (10 ** np.linspace(0, 1, len(colors_temp)) - 1) / 9
    else:
        raise ValueError("Unexpected colorscale transform")
    return [(v, clr) for v, clr in zip(scale_values, colors_temp)]


def build_datashader_plot(
        df, aggregate, aggregate_column, new_coordinates, position,
        x_range, y_range, df_hospitals, df_covid, df_acs2018, population_enabled,
        population_colorscale, hospital_enabled, hospital_colorscale,
        covid_enabled, covid_count_type, colorscale_transform
):
    """
    Build choropleth figure

    Args:


    Returns:
        Choropleth figure dictionary
    """
    map_graph = {
        'data': [],
        'layout': {
                'template': template,
                'uirevision': True,
                'mapbox': {
                    'style': "dark",
                    'accesstoken': token,
                },
                'margin': {"r": 140, "t": 0, "l": 0, "b": 0},
                'height': 500,
                'shapes': [{
                    'type': 'rect',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x0': 0,
                    'y0': 0,
                    'x1': 1,
                    'y1': 1,
                    'line': {
                        'width': 1,
                        'color': '#191a1a',
                    }
                }],
                'showlegend': False
            }
    }
    if population_enabled:
        global data_3857, data_center_3857, data_4326, data_center_4326

        x0, x1 = x_range
        y0, y1 = y_range

        # Build query expressions
        query_expr_xy = f"(x >= {x0}) & (x <= {x1}) & (y >= {y0}) & (y <= {y1})"
        datashader_color_scale = {}

        if population_colorscale == 'Purblue':
            aggregate_column = 'age'
            datashader_color_scale['color_key'] = colors[aggregate_column] 
        else:
            aggregate_column = 'x'
            aggregate = 'count'
            datashader_color_scale['cmap'] = [i[1] for i in build_colorscale(population_colorscale, colorscale_transform, aggregate, aggregate_column)]
            if not isinstance(df, cudf.DataFrame):
                df[aggregate_column] = df[aggregate_column].astype('int8')

        cvs = ds.Canvas(
            plot_width=2400,
            plot_height=1200,
            x_range=x_range, y_range=y_range
        )
        agg = cvs.points(
            df, x='x', y='y', agg=getattr(ds, aggregate)(aggregate_column)
        )

        cmin = cupy.asnumpy(agg.min().data)
        cmax = cupy.asnumpy(agg.max().data)
        # Count the number of selected people
        temp = agg.sum()
        temp.data = cupy.asnumpy(temp.data)
        n_selected = int(temp)
        layers = []
        if n_selected == 0:
            # Nothing to display
            lat = [None]
            lon = [None]
            customdata = [None]
            marker = {}
            layers = []
        else:
            # Shade aggregation into an image that we can add to the map as a mapbox
            # image layer
            img = tf.shade(agg, how='log',**datashader_color_scale).to_pil()


            # Add image as mapbox image layer. Note that as of version 4.4, plotly will
            # automatically convert the PIL image object into a base64 encoded png string
            layers = [
                {
                    "sourcetype": "image",
                    "source": img,
                    "coordinates": new_coordinates
                }
            ]

            # Do not display any mapbox markers
            lat = [None]
            lon = [None]
            customdata = [None]

        if aggregate == 'count_cat':
            # for `Age By PurBlue` category
            colorscale = (10 ** np.linspace(0, 1, len(colors['age'])) - 1) / 9
            marker = dict(
                    size=0,
                    showscale=True,
                    colorbar= {"title": {
                        "text": 'Age', "side": "right", "font": {"size": 14}
                    }},
                    colorscale=[(v, clr) for v, clr in zip(colorscale, colors['age'])],
                    cmin=0,
                    cmax=65,
                )

            map_graph['data'].append(
                {
                    'type': 'scattermapbox',
                    'lat': lat, 'lon': lon,
                    'customdata': customdata,
                    'marker': marker,
                    'hoverinfo': 'none',
                }
            )
            map_graph['layout']['annotations'] = []
        else:
            marker = dict(
                    size=0,
                    showscale=True,
                    colorbar= {"title": {
                        "text": 'Population', "side": "right", "font": {"size": 14}
                    }},
                    colorscale=build_colorscale(
                        population_colorscale, colorscale_transform,
                        aggregate, aggregate_column
                    ),
                    cmin=cmin,
                    cmax=cmax
                )
            map_graph['data'].append(
                {
                    'type': 'scattermapbox',
                    'lat': lat, 'lon': lon,
                    'customdata': customdata,
                    'marker': marker,
                    'hoverinfo': 'none'
                }
            )
        # Build map figure
        

        map_graph['layout']['mapbox'].update({'layers': layers})
    map_graph['layout']['mapbox'].update(position)

    if df_hospitals is not None and isinstance(df_hospitals, pd.DataFrame):
        map_graph['data'].append(
             {
                'type': 'scattermapbox',
                'lat': df_hospitals.Y.values,
                'lon': df_hospitals.X.values,
                'marker': go.scattermapbox.Marker(
                    size=df_hospitals.BEDS_sizes.values + 250,
                    sizemin=2,
                    color='black',
                    opacity=0.9,
                    sizeref=120,
                ),
                'hoverinfo': 'none'
            }
        )
        map_graph['data'].append({
                'type': 'scattermapbox',
                'lat': df_hospitals.Y.values,
                'lon': df_hospitals.X.values,
                'marker': go.scattermapbox.Marker(
                    size=df_hospitals.BEDS_sizes.values,
                    sizemin=2,
                    color=hospital_colorscale,
                    opacity=0.9,
                    sizeref=120,
                ),
                'hovertemplate': (
                    '<b>%{hovertext}</b><br><br>BEDS = %{text}<extra></extra>'
                ),
                'text': df_hospitals.BEDS,
                'hovertext': df_hospitals.NAME,
                'mode': 'markers',
                'showlegend': False,
                'subplot': 'mapbox'
            }
        )
    if df_covid is not None:
        df_covid_yesterday = df_covid[2]
        yesterday = df_covid_yesterday.Last_Update.to_pandas().max().strftime("%B, %d")
        df_covid = df_covid[1]
        today = df_covid.Last_Update.to_pandas().max().strftime("%B, %d")
        size_markers = np.copy(df_covid.Confirmed.to_array())
        size_markers_labels = np.copy(size_markers)
        annotations = {
                'text': size_markers_labels
        }
        if covid_count_type == 0:
            size_markers[size_markers <= 2] = 2
            factor = 'Confirmed Cases as of '+today+' = %{text}'
            sizeref = 120
            marker_border = 250
        if covid_count_type == 1:
            size_markers_yesterday = np.copy(df_covid_yesterday.Confirmed.to_array())
            size_markers_yesterday[size_markers_yesterday == 0] = 1
            size_markers = (np.nan_to_num((size_markers - df_covid_yesterday.Confirmed.to_array())*100/size_markers_yesterday)).astype('int64')
            size_markers_labels = np.copy(size_markers)
            size_markers = np.absolute(size_markers)
            annotations = {
                'text': size_markers_labels,
                'customdata': np.vstack([df_covid_yesterday.Confirmed.to_array(), df_covid.Confirmed.to_array()]).T
            }
            factor = 'Confirmed cases:<br> '+yesterday+': %{customdata[0]}<br> '+today+': %{customdata[1]}<br> % increase = %{text}'
            sizeref = 15
            marker_border = 32
        elif covid_count_type == 2:
            df_covid = df_covid.merge(df_acs2018, on='COUNTY')
            size_markers = df_covid.Confirmed.to_array()
            size_markers = np.nan_to_num(size_markers/(df_covid.acs2018_population.to_array()/1000)).astype('float')
            size_markers_labels = np.around(size_markers, 2)
            annotations = {
                'text': size_markers_labels
            }
            factor = '<i>sourced from LATEST 2018 census projection </i> <br>No. of cases / 1000 people = %{text}'
            sizeref = 5/(size_markers.max())
            marker_border = 2*sizeref

        size_markers[size_markers >= np.percentile(size_markers, 99.9)] = np.percentile(size_markers, 99.9)

        map_graph['data'].append(
             {
                'type': 'scattermapbox',
                'lat': df_covid.Lat.to_array(),
                'lon': df_covid.Long_.to_array(),
                'marker': go.scattermapbox.Marker(
                    size=size_markers + marker_border,
                    # sizemin=2,
                    color='black',
                    opacity=0.6,
                    sizeref=sizeref,
                ),
                'hoverinfo': 'none'
            }
        )
        map_graph['data'].append({
                'type': 'scattermapbox',
                'lat': df_covid.Lat.to_array(),
                'lon': df_covid.Long_.to_array(),
                'marker': go.scattermapbox.Marker(
                    size=size_markers,
                    sizemin=2,
                    color='#f20e5a',
                    opacity=0.6,
                    sizeref=sizeref,
                ),
                'hovertemplate': (
                    '<b>%{hovertext}</b><br><br>'+factor+'<extra></extra>'
                ),
                **annotations,
                'hovertext': df_covid.COUNTY,
                'mode': 'markers',
                'showlegend': False,
                'subplot': 'mapbox'
            }
        )


    return map_graph


def build_histogram_default_bins(
        df, column, selections, query_cache, orientation
):
    """
    Build histogram figure

    Args:
        df: pandas or cudf DataFrame
        column: Column name to build histogram from
        selections: Dictionary from column names to query expressions
        query_cache: Dict from query expression to filtered DataFrames

    Returns:
        Histogram figure dictionary
    """
    bin_edges = df.index.values
    counts = df.values

    range_ages = [20, 40, 60, 84, 85]
    colors_ages = ['#C700E5','#9D00DB', '#7300D2','#4900C8','#1F00BF']
    labels = ['0-20', '21-40', '41-60', '60-84', '85+']
    # centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    if orientation == 'h':
        fig = {
            'data':[],
            'layout': {
                'xaxis': {
                    'title': {
                        'text': "Count"
                    }
                },
                'selectdirection': 'v',
                'dragmode': 'select',
                'template': template,
                'uirevision': True,
            }
        }
    else:
        fig = {
            'data':[],
            'layout': {
                'yaxis': {
                    'title': {
                        'text': "Count"
                    }
                },
                'selectdirection': 'h',
                'dragmode': 'select',
                'template': template,
                'uirevision': True,
            }
        }

    for index, ages in enumerate(range_ages):
        if index == 0:
            count_temp = counts[:ages]
            bin_edges_temp = bin_edges[:ages]
        else:
            count_temp = counts[range_ages[index-1]:ages]
            bin_edges_temp = bin_edges[range_ages[index-1]:ages]

        if orientation == 'h':
            fig['data'].append(
                {
                    'type': 'bar', 'x': count_temp, 'y': bin_edges_temp,
                    'marker': {'color': colors_ages[index]},
                    'name': labels[index]
                }
            )
        else:
            fig['data'].append(
                {
                    'type': 'bar', 'x': bin_edges_temp, 'y': count_temp,
                    'marker': {'color': colors_ages[index]},
                    'name': labels[index]
                }
            )

    if column not in selections:
        for i in range(len(fig['data'])):
            fig['data'][i]['selectedpoints'] = False

    return fig

def _custom_reset_index(df):
    #reduces gpu usage by around 2GB
    df.index = cudf.core.RangeIndex(0, len(df), name=df.index)
    return df

def build_updated_figures(
        df, df_hospitals, df_covid, df_acs2018, relayout_data,
        aggregate, population_enabled, population_colorscale,
        hospital_enabled, hospital_colorscale,
        covid_enabled, covid_count_type,
        data_3857, data_center_3857, data_4326, data_center_4326
):
    """
    Build all figures for dashboard

    Returns:
        tuple of figures in the following order
        (relayout_data, age_male_histogram, age_female_histogram,
        n_selected_indicator)
    """

    colorscale_transform, aggregate_column = 'linear', 'age'

    drop_down_queries = ''

    # if relayout_data is not None:
    transformer_4326_to_3857 = Transformer.from_crs("epsg:4326", "epsg:3857")

    def epsg_4326_to_3857(coords):
        return [
            transformer_4326_to_3857.transform(*reversed(row)) for row in coords
        ]

    coordinates_4326 = relayout_data and relayout_data.get('mapbox._derived', {}).get('coordinates', None)

    if coordinates_4326:
        lons, lats = zip(*coordinates_4326)
        lon0, lon1 = max(min(lons), data_4326[0][0]), min(max(lons), data_4326[1][0])
        lat0, lat1 = max(min(lats), data_4326[0][1]), min(max(lats), data_4326[1][1])
        coordinates_4326 = [
            [lon0, lat0],
            [lon1, lat1],
        ]
        coordinates_3857 = epsg_4326_to_3857(coordinates_4326)

        position = {
            'zoom': relayout_data.get('mapbox.zoom', None),
            'center': relayout_data.get('mapbox.center', None)
        }
    else:
        position = {
            'zoom': 2.5,
            'pitch': 0,
            'bearing': 0,
            'center': {'lon': data_center_4326[0][0]-100, 'lat': data_center_4326[0][1]-10}
        }
        coordinates_3857 = data_3857
        coordinates_4326 = data_4326

    new_coordinates = [
        [coordinates_4326[0][0], coordinates_4326[1][1]],
        [coordinates_4326[1][0], coordinates_4326[1][1]],
        [coordinates_4326[1][0], coordinates_4326[0][1]],
        [coordinates_4326[0][0], coordinates_4326[0][1]],
    ]

    x_range, y_range = zip(*coordinates_3857)
    x0, x1 = x_range
    y0, y1 = y_range

    # Build query expressions

    mask_ = (df['x']>=x0) & (df['x']<=x1) & (df['y']>=y0) & (df['y']<=y1)
    if(mask_.sum() != len(df)):
        df_map = df[mask_]
        df_map.index = cudf.core.RangeIndex(0,len(df_map))
    else:
        df_map = df
    
    del(mask_)
    
    datashader_plot = build_datashader_plot(
        df, aggregate,
        aggregate_column, new_coordinates, position, x_range, y_range,
        df_hospitals, df_covid, df_acs2018, population_enabled, population_colorscale,
        hospital_enabled, hospital_colorscale,
        covid_enabled, covid_count_type,
        colorscale_transform
    )


    # Build indicator figure
    n_selected_indicator = {
        'data': [{
            'title': {"text": "Visible Population"},
            'type': 'indicator',
            'value': len(
                df_map
            ),
            'number': {
                'font': {
                    'color': text_color,
                    'size': '40px'
                },
                "valueformat": ","
            }
        },
        ],
        'layout': {
            'template': template,
            'height': row_heights[5],
            'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10}
        }
    }
    n_selected_indicator['data'][0]['domain'] =  {'x': [0, 1], 'y': [0, 0.5]}

    if df_covid is not None:
        n_selected_indicator['data'][0]['domain'] =  {'x': [0, 0.5], 'y': [0, 0.5]}
        x_range, y_range = zip(*coordinates_4326)
        x0, x1 = x_range
        y0, y1 = y_range
        query_expr_xy_hosp = f"(Long_ >= {x0}) & (Long_ <= {x1}) & (Lat >= {y0}) & (Lat <= {y1})"
        df_covid = df_covid[1].query(query_expr_xy_hosp).reset_index(drop=True)
        n_selected_indicator['data'].append({
            'title': {"text": "Visible Total Cases"},
            'type': 'indicator',
            'value': df_covid.Confirmed.sum(),
            'domain': {'x': [0.51, 1], 'y': [0, 0.5]},
            'number': {
                'font': {
                    'color': text_color,
                    'size': '40px'
                },
                "valueformat": ","
            }
        })

    if df_hospitals is not None:
        n_selected_indicator['data'][0]['domain'] =  {'x': [0, 0.35], 'y': [0, 0.5]}
        domain_1 = {'x': [0.36, 0.7], 'y': [0, 0.5]}
        domain_2 = {'x': [0.71, 1], 'y': [0, 0.5]}
        if len(n_selected_indicator['data']) == 2:
            n_selected_indicator['data'][0]['domain'] =  {'x': [0, 0.25], 'y': [0, 0.5]}
            n_selected_indicator['data'][1]['domain'] =  {'x': [0.26, 0.5], 'y': [0, 0.5]}
            domain_1 = {'x': [0.51, 0.75], 'y': [0, 0.5]}
            domain_2 = {'x': [0.76, 1], 'y': [0, 0.5]}
            
        x_range, y_range = zip(*coordinates_4326)
        x0, x1 = x_range
        y0, y1 = y_range
        query_expr_xy_hosp = f"(X >= {x0}) & (X <= {x1}) & (Y >= {y0}) & (Y <= {y1})"
        df_hospitals = df_hospitals.query(query_expr_xy_hosp).reset_index(drop=True)
        n_selected_indicator['data'].append({
            'title': {"text": "Known Hospital Beds"},
            'type': 'indicator',
            'value': df_hospitals.BEDS.sum(),
            'domain': domain_1,
            'number': {
                'font': {
                    'color': text_color,
                    'size': '40px'
                },
                "valueformat": ","
            }
        })
        n_selected_indicator['data'].append({
            'title': {"text": "People to Beds"},
            'type': 'indicator',
            'value': round(len(df_map)/df_hospitals.BEDS.sum()),
            'domain': domain_2,
            'number': {
                'font': {
                    'color': text_color,
                    'size': '40px'
                },
                "valueformat": ","
            }
        })

    query_cache = {}

    return (datashader_plot, n_selected_indicator)

def generate_covid_bar_plots(df, scale_covid, category_covid):
    df = df[0]
    states = df.Province_State.unique().tolist()
    for rem_state in ['Wuhan Evacuee','Diamond Princess','Recovered', 'Grand Princess']:
        if rem_state in states:
            states.remove(rem_state)

    fig = make_subplots(rows=12, cols=5, subplot_titles=states)
    index = 0
    for state in states:
        df_temp = df.query('Province_State == @state').reset_index(drop=True)
        if(category_covid == 'Total cases'):
            fig.add_trace(go.Scatter(x=df_temp.Last_Update, y=df_temp.Confirmed, name='Confirmed', marker=dict(color='#c724e5'), hoverinfo='text', hovertemplate='Confirmed=%{text}<extra></extra>', text=df_temp.Confirmed), row=int(index/5) + 1, col=int(index%5)+1)
        else:
            fig.add_trace(go.Scatter(x=df_temp.Last_Update, y=df_temp.Deaths, name='Deaths', marker=dict(color='#b7b7b7'), hoverinfo='text', hovertemplate='Deaths=%{text}<extra></extra>', text=df_temp.Deaths), row=int(index/5) + 1, col=int(index%5)+1)
        
        fig.update_yaxes(rangemode='tozero',autorange=True, type=scale_covid, row=int(index/5) + 1, col=int(index%5)+1)
        index += 1
    fig.update_layout(
        template=template,
        height= row_heights[3],
        showlegend=False,
    )
    return fig

def get_covid_bar_chart(df, scale_covid, category_covid):
    global cache_fig_1, cache_fig_2, cache_fig_3, cache_fig_4

    if scale_covid == 'linear' and category_covid == 'Total cases':
        if cache_fig_1 is None:
            cache_fig_1 = generate_covid_bar_plots(df, scale_covid, category_covid)
        return cache_fig_1
    elif scale_covid == 'linear' and category_covid == 'Total deaths':
        if cache_fig_2 is None:
            cache_fig_2 = generate_covid_bar_plots(df, scale_covid, category_covid)
        return cache_fig_2
    elif scale_covid == 'log' and category_covid == 'Total cases':
        if cache_fig_3 is None:
            cache_fig_3 = generate_covid_bar_plots(df, scale_covid, category_covid)
        return cache_fig_3
    else:
        if cache_fig_4 is None:
            cache_fig_4 = generate_covid_bar_plots(df, scale_covid, category_covid)
        return cache_fig_4

    
    


def generate_covid_charts(client):
    @app.callback(
        [Output('covid-histogram', 'figure')],
        [Input('covid-histogram', 'selectedData'), Input('scale-covid-dropdown', 'value'),
        Input('category-covid-dropdown', 'value')]
    )
    def update_covid(selected_covid, scale_covid, category_covid):
        df = client.get_dataset('pd_covid')
        figure = delayed(get_covid_bar_chart)(df, scale_covid, category_covid)
        (figure_) = figure.compute()
        return [figure_]

def register_update_plots_callback(client):
    """
    Register Dash callback that updates all plots in response to selection events
    Args:
        df_d: Dask.delayed pandas or cudf DataFrame
    """
    @app.callback(
        [Output('indicator-graph', 'figure'), Output('map-graph', 'figure'),
         ],
        [Input('map-graph', 'relayoutData'),
            Input('population-toggle', 'on'), Input('colorscale-dropdown-population', 'value'), 
            Input('hospital-toggle', 'on'), Input('colorscale-dropdown-hospital', 'value'),
            Input('covid-toggle', 'on'), Input('covid_count_type', 'value'),
        ]
    )
    def update_plots(
            relayout_data,
            population_enabled, population_colorscale,
            hospital_enabled, hospital_colorscale,
            covid_enabled, covid_count_type,
    ):
        global data_3857, data_center_3857, data_4326, data_center_4326, covid_last_update_time

        if int(time.time() - covid_last_update_time) > 21600:
            # update covid data every six hours
            update_covid_data(client)
            covid_last_update_time = time.time()
            covid_last_update_date = datetime.datetime.today().strftime("%b-%d-%Y at %H:%M")


        t0 = time.time()

        df_d = client.get_dataset('c_df_d')
        df_hospitals = None
        df_covid = None
        df_acs2018 = None

        if covid_count_type == 2:
            df_acs2018 = client.get_dataset('c_acs_2018')
        
        if hospital_enabled:
            df_hospitals = client.get_dataset('pd_hospitals')
        
        if covid_enabled:
            df_covid = client.get_dataset('pd_covid')

        if data_3857 == []:
            projections = delayed(set_projection_bounds)(df_d)
            data_3857, data_center_3857, data_4326, data_center_4326 = projections.compute()

        figures_d = delayed(build_updated_figures)(
            df_d, df_hospitals, df_covid, df_acs2018, relayout_data, 'count_cat',
            population_enabled, population_colorscale,
            hospital_enabled, hospital_colorscale,
            covid_enabled, covid_count_type,
            data_3857, data_center_3857, data_4326, data_center_4326
        )

        figures = figures_d.compute()

        (datashader_plot, n_selected_indicator) = figures

        print(f"Update time: {time.time() - t0}")
        return (
            n_selected_indicator, datashader_plot
        )

def check_dataset(dataset_url, data_path):
    if not os.path.exists(data_path):
        print(f"Dataset not found at "+data_path+".\n"
              f"Downloading from {dataset_url}")
        # Download dataset to data directory
        os.makedirs('../data', exist_ok=True)
        data_gz_path = data_path.split('/*')[0] + '.tar.gz'
        with requests.get(dataset_url, stream=True) as r:
            r.raise_for_status()
            with open(data_gz_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Decompressing...")
        f_in = tarfile.open(data_gz_path, 'r:gz')
        f_in.extractall('../data')

        print("Deleting compressed file...")
        os.remove(data_gz_path)

        print('done!')
    else:
        print(f"Found dataset at {data_path}")

def update_covid_data(client):
    global cache_fig_1, cache_fig_2, cache_fig_3, cache_fig_4
    covid_data_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/%s.csv'
    print("Updating Covid data")
    cache_fig_1, cache_fig_2, cache_fig_3, cache_fig_4 = None, None, None, None            
    pd_covid = delayed(load_covid)(covid_data_path).persist()
    client.unpublish_dataset('pd_covid')
    client.publish_dataset(pd_covid=pd_covid)

def publish_dataset_to_cluster():
    census_data_url = 'https://s3.us-east-2.amazonaws.com/rapidsai-data/viz-data/census_data_minimized.parquet.tar.gz'
    census_data_path = "../data/census_data_minimized.parquet"
    check_dataset(census_data_url, census_data_path)
    
    acs2018_data_path = "../data/acs2018_county_population.parquet"

    hospital_path = '../data/Hospitals.csv'
    covid_data_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/%s.csv'
    # Note: The creation of a Dask LocalCluster must happen inside the `__main__` block,
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0")
    client = Client(cluster)
    print(f"Dask status: {cluster.dashboard_link}")

    # Load dataset and persist dataset on cluster
    def load_and_publish_dataset():
        # cudf DataFrame
        c_df_d = delayed(load_dataset)(census_data_path).persist()
        # pandas DataFrame
        # pd_df_d = delayed(c_df_d.to_pandas)().persist()
        pd_covid = delayed(load_covid)(covid_data_path).persist()
        pd_hospitals = delayed(load_hospitals)(hospital_path).persist()
        c_acs_2018 = delayed(load_dataset(acs2018_data_path)).persist()

        # Unpublish datasets if present
        for ds_name in ['pd_df_d', 'c_df_d', 'pd_states_last_5_days', 'pd_states_today', 'pd_hospitals', 'c_acs_2018']:
            if ds_name in client.datasets:
                client.unpublish_dataset(ds_name)

        # Publish datasets to the cluster
        # client.publish_dataset(pd_df_d=pd_df_d)
        client.publish_dataset(c_df_d=c_df_d)
        client.publish_dataset(pd_covid=pd_covid)
        client.publish_dataset(c_acs_2018=c_acs_2018)
        client.publish_dataset(pd_hospitals=pd_hospitals)

    load_and_publish_dataset()

    # Precompute field bounds
    c_df_d = client.get_dataset('c_df_d')

    # Register top-level callback that updates plots
    register_update_plots_callback(client)
    generate_covid_charts(client)


def server():
    # gunicorn entry point when called with `gunicorn 'app:server()'`
    publish_dataset_to_cluster()
    return app.server


if __name__ == '__main__':
    # development entry point
    publish_dataset_to_cluster()

    # Launch dashboard
    app.run_server(debug=False, dev_tools_silence_routes_logging=True, host='0.0.0.0')