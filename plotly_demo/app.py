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
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq
from plotly.colors import sequential
from pyproj import Transformer

from dask import delayed
from distributed import Client
from dask_cuda import LocalCUDACluster
import plotly.graph_objects as go

import cudf
import cupy

# Disable cupy memory pool so that cupy immediately releases GPU memory
cupy.cuda.set_allocator(None)

# Colors
bgcolor = "#191a1a"  # mapbox dark map land color
text_color = "#cfd8dc"  # Material blue-grey 100
mapbox_land_color = "#343332"

# Figure template
row_heights = [150, 440, 200]
template = {
    'layout': {
        'paper_bgcolor': bgcolor,
        'plot_bgcolor': bgcolor,
        'font': {'color': text_color},
        "margin": {"r": 0, "t": 30, "l": 0, "b": 20},
        'bargap': 0.05,
        'xaxis': {'showgrid': False, 'automargin': True},
        'yaxis': {'showgrid': True, 'automargin': True},
                #   'gridwidth': 0.5, 'gridcolor': mapbox_land_color},
    }
}


colors = {}
mappings = {}
colors['sex'] = ['#0000FF', '#00ff00']
colors['income'] = [
    "#e0bb7f",
    "#00b3b3",
    "#ff748c",
    "#a280ff",
    "#ff8b61",
    "#bd8ad4",
    "#ffc874",
    "#fff968",
    "#93b6ff",
    "#5a1df4",
    "#0e3c17",
    "#d1c9bd",
    "#8cacc0",
    "#93a753",
    "#bada55",
    "#d1c9bd",
    "#d1c9bd",
    "#707372",
    "#e4002b",
    "#403a60",
    "#975c9f",
 ]
colors['education'] = [
    "#e0bb7f",
    "#00b3b3",
    "#ff748c",
    "#a280ff",
    "#ff8b61",
    "#bd8ad4",
    "#ffc874",
    "#fff968",
    "#93b6ff",
    "#5a1df4",
    "#0e3c17",
    "#d1c9bd",
    "#8cacc0",
    "#93a753",
    "#bada55",
    "#d1c9bd",
    "#d1c9bd",
]

colors['cow']= [
    "#e0bb7f",
    "#00b3b3",
    "#ff748c",
    "#a280ff",
    "#ff8b61",
    "#bd8ad4",
    "#ffc874",
    "#fff968",
]
# Load mapbox token from environment variable or file
token = os.getenv('MAPBOX_TOKEN')
if not token:
    token = open(".mapbox_token").read()


# Names of float columns
float_columns = [
    'cow', 'education', 'x', 'y', 'income', 'age', 'sex'
]

column_labels = {
    # 'education': 'education',
    'sex': 'sex',
    # 'income': 'income',
    # 'cow': 'Class of Worker',
}

cow_mappings_hover = {
    0: "Private for-profit wage and salary workers: Employee of private company workers",
    1: "Private for-profit wage and salary workers: Self-employed in own incorporated business workers",
    2: "Private not-for-profit wage and salary workers",
    3: "Local government workers",
    4: "State government workers",
    5: "Federal government workers",
    6: "Self-employed in own not incorporated business workers",
    7: "Unpaid family workers",
    8: "Data not available/under 16 years",
}

mappings['cow'] = {
    -1: "All",
    0: "Emp private for-profit",
    1: "Self-employed for-profit",
    2: "Emp private not-for-profit",
    3: "Local gov emp",
    4: "State gov emp",
    5: "Federal gov emp",
    6: "Self-emp in own not business",
    7: "Unpaid family workers",
    8: "Data not available/under 16 years",
}

mappings['sex'] = {
    -1: "All genders",
    0: 'Males',
    1: 'Females'
}

mappings['education']= {
    0: "No schooling completed",
    1: "Nursery to 4th grade",
    2: "5th and 6th grade",
    3: "7th and 8th grade",
    4: "9th grade",
    5: "10th grade",
    6: "11th grade",
    7: "12th grade, no diploma",
    8: "High school graduate, GED, or alternative",
    9: "Some college, less than 1 year",
    10: "Some college, 1 or more years, no degree",
    11: "Associate's degree",
    12: "Bachelor's degree",
    13: "Master's degree",
    14: "Professional school degree",
    15: "Doctorate degree",
    16: 'below 16 years/data not available'
}

mappings['income'] = {
    0: "$1 to $2,499 or loss",
    1: "$2,500 to $4,999",
    2: "$5,000 to $7,499",
    3: "$7,500 to $9,999",
    4: "$10,000 to $12,499",
    5: "$12,500 to $14,999",
    6: "$15,000 to $17,499",
    7: "$17,500 to $19,999",
    8: "$20,000 to $22,499",
    9: "$22,500 to $24,999",
    10: "$25,000 to $29,999",
    11: "$30,000 to $34,999",
    12: "$35,000 to $39,999",
    13: "$40,000 to $44,999",
    14: "$45,000 to $49,999",
    15: "$50,000 to $54,999",
    16: "$55,000 to $64,999",
    17: "$65,000 to $74,999",
    18: "$75,000 to $99,999",
    19: "$100,000 or more",
    20: 'below 25 years/unknown',
}

data_center_3857, data_3857, data_4326, data_center_4326 = [], [], [], []

def load_dataset(path):
    """
    Args:
        path: Path to arrow file containing mortgage dataset

    Returns:
        pandas DataFrame
    """
    df_d = cudf.read_parquet(path)
    df_d['sex'] = df_d.sex.to_pandas().astype('category')
    print(df_d.dtypes)
    return df_d


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
            'Census Data',
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
                html.Div(children=[
                    html.Button(
                        "Clear All Selections", id='clear-all', className='reset-button'
                    ),
                ]),
                html.H4([
                    "Selected Polpulation",
                ], className="container_title"),
                dcc.Loading(
                    dcc.Graph(
                        id='indicator-graph',
                        figure=blank_fig(row_heights[0]),
                        config={'displayModeBar': False},
                    ),
                    style={'height': row_heights[0]},
                )
            ], className='six columns pretty_container', id="indicator-div"),
            html.Div(children=[
                html.H4([
                    "Configuration",
                ], className="container_title"),
                html.Table([
                    html.Col(style={'width': '100px'}),
                    html.Col(),
                    html.Col(),
                    html.Tr([
                        html.Td(
                            html.Div("GPU"), className="config-label"
                        ),
                        html.Td(daq.DarkThemeProvider(daq.BooleanSwitch(
                            on=True,
                            color='#00cc96',
                            id='gpu-toggle',
                        ))),
                        html.Td(html.Button(
                            "Reset GPU", id='reset-gpu', style={'width': '100%'}
                        )),
                        html.Div(id='reset-gpu-complete', style={'display': 'hidden'})
                    ]),
                    html.Tr([
                        html.Td(html.Div("Color by"), className="config-label"),
                        html.Td(dcc.Dropdown(
                            id='aggregate-dropdown',
                            options=
                            [
                                {'label': 'count', 'value': 'count'},
                                {'label': 'category by gender', 'value': 'count_cat'},
                            ],
                            value='count',
                            searchable=False,
                            clearable=False,
                        )),
                        html.Td(dcc.Dropdown(
                            id='colorscale-dropdown',
                            options=[
                                {'label': cs, 'value': cs}
                                for cs in ['Viridis', 'Cividis', 'Inferno', 'Magma', 'Plasma']
                            ],
                            value='Viridis',
                            searchable=False,
                            clearable=False,
                        )),
                    ]),
                    html.Tr([
                        html.Td(html.Div("Class of Workers"), className="config-label"),
                        html.Td(dcc.Dropdown(
                            id='cow-dropdown',
                            options=[
                                {'label': val, 'value': key}
                                for key, val in mappings['cow'].items()
                            ],
                            value=-1,
                            searchable=False,
                            clearable=False,
                        )),
                        html.Td(dcc.Dropdown(
                            id='sex-dropdown',
                            options=[
                                {'label': val, 'value': key}
                                for key, val in mappings['sex'].items()
                            ],
                            value=-1,
                            searchable=False,
                            clearable=False,
                        )),
                        
                        
                    ]),
                ], style={'width': '100%', 'height': f'{row_heights[0]}px'}),
            ], className='six columns pretty_container', id="config-div"),
        ]),
        html.Div(children=[
            html.Button("Clear Selection", id='reset-map', className='reset-button'),
            html.H4([
                "US Population(each individual)",
            ], className="container_title"),
            dcc.Graph(
                id='map-graph',
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
            html.Button("Clear Selection", id='reset-scatter', className='reset-button'),
            html.H4([
                "Education - Income distribution",
            ], className="container_title"),
            dcc.Graph(
                id='scatter-graph',
                figure=blank_fig(row_heights[1]),
            ),
            
        ], className='twelve columns pretty_container',
            style={
                'width': '98%',
                'margin-right': '0',
            },
            id="scatter-div"
        ),
        html.Div(children=[
            html.Div(
                children=[
                    html.Button(
                        "Clear Selection", id='clear-age', className='reset-button'
                    ),
                    html.H4([
                        "Age",
                    ], className="container_title"),
                    
                    dcc.Graph(
                        id='age-histogram',
                        config={'displayModeBar': False},
                        figure=blank_fig(row_heights[1]),
                        animate=True
                    ),
                ],
                className='twelve columns pretty_container', id="age-div"
            )
        ]),        
    ]),
    html.Div(
        [
            html.H4('Acknowledgements', style={"margin-top": "0"}),
            dcc.Markdown('''\
 - Dashboard written in Python using the [Dash](https://dash.plot.ly/) web framework.
 - GPU accelerated provided by the [cudf](https://github.com/rapidsai/cudf) and
 [cupy](https://cupy.chainer.org/) libraries.
 - Base map layer is the ["dark" map style](https://www.mapbox.com/maps/light-dark/)
 provided by [mapbox](https://www.mapbox.com/).
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

# # Register callbacks
# @app.callback(
#     [Output('aggregate-col-dropdown', 'options'),
#      Output('aggregate-col-dropdown', 'disabled')],
#     [Input('sex-dropdown', 'value')]
# )
# def update_agg_col_dropdown(agg):
#     if agg == 'count':
#         options = [{'label': 'NA',
#                     'value': 'NA'}]
#         disabled = True
#     else:
#         options = [{'label': v, 'value': k} for k, v in column_labels.items()]
#         disabled = False
#     return options, disabled


# Clear/reset button callbacks
@app.callback(
    Output('map-graph', 'relayoutData'),
    [Input('reset-map', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_map(*args):
    return None

@app.callback(
    Output('age-histogram', 'selectedData'),
    [Input('clear-age', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_age_hist_selections(*args):
    return None

@app.callback(
    Output('scatter-graph', 'selectedData'),
    [Input('reset-scatter', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_scatter_selections(*args):
    return None

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
        df, aggregate, aggregate_column, colorscale_name, colorscale_transform,
        new_coordinates, position, x_range, y_range
):
    """
    Build choropleth figure

    Args:
        df: pandas or cudf DataFrame
        aggregate: Aggregate operation (count, mean, etc.)
        aggregate_column: Column to perform aggregate on. Ignored for 'count' aggregate
        colorscale_name: Name of plotly colorscale
        colorscale_transform: Colorscale transformation
        clear_selection: If true, clear choropleth selection. Otherwise leave
            selection unchanged

    Returns:
        Choropleth figure dictionary
    """

    global data_3857, data_center_3857, data_4326, data_center_4326

    x0, x1 = x_range
    y0, y1 = y_range

    # Build query expressions
    query_expr_xy = f"(x >= {x0}) & (x <= {x1}) & (y >= {y0}) & (y <= {y1})"
    datashader_color_scale = {}

    if aggregate == 'count_cat':
        datashader_color_scale['color_key'] = colors[aggregate_column] 
    else:
        datashader_color_scale['cmap'] = [i[1] for i in build_colorscale(colorscale_name, colorscale_transform, aggregate, aggregate_column)]
        if not isinstance(df, cudf.DataFrame):
            df[aggregate_column] = df[aggregate_column].astype('int8')

    print(df.dtypes)
    cvs = ds.Canvas(
        plot_width=1400,
        plot_height=1400,
        x_range=x_range, y_range=y_range
    )
    
    agg = cvs.points(
        df, x='x', y='y', agg=getattr(ds, aggregate)(aggregate_column)
    )
    

    # Count the number of selected towers
    temp = agg.sum()
    temp.data = cupy.asnumpy(temp.data)
    n_selected = int(temp)
    
    if n_selected == 0:
        # Nothing to display
        lat = [None]
        lon = [None]
        customdata = [None]
        marker = {}
        layers = []
    # elif n_selected < 5000:
    #     # Display each individual point using a scattermapbox trace. This way we can
    #     # give each individual point a tooltip

    #     ddf_gpu_small_expr = ' & '.join(
    #         [query_expr_xy]
    #     )
    #     ddf_gpu_small = df.query(ddf_gpu_small_expr).to_pandas()

    #     x, y, sex, edu, inc, cow = (
    #         ddf_gpu_small.x, ddf_gpu_small.y, ddf_gpu_small.sex, ddf_gpu_small.education, ddf_gpu_small.income, ddf_gpu_small.cow
    #     )

    #     # Format creation date column for tooltip
    #     # created = pd.to_datetime(created.tolist()).strftime('%x')


    #     # Build array of the integer category codes to use as the numeric color array
    #     # for the scattermapbox trace
    #     sex_codes = sex.unique().tolist()

    #     # Build marker properties dict
    #     marker = {
    #         'color': sex_codes,
    #         'colorscale': colors[aggregate_column],
    #         'cmin': 0,
    #         'cmax': 3,
    #         'size': 5,
    #         'opacity': 0.6,
    #     }
    #     lat = list(zip(
    #         x.astype(str)
    #     ))
    #     lon = list(zip(
    #         y.astype(str)
    #     ))
    #     customdata = list(zip(
    #         sex.astype(str),
    #         edu.astype(str),
    #         inc.astype(str),
    #         cow.astype(str)
    #     ))
    #     layers = []
    else:
        # Shade aggregation into an image that we can add to the map as a mapbox
        # image layer
        max_px = 1
        if n_selected<5000:
            max_px=10
        img = tf.shade(agg, **datashader_color_scale)
        img = tf.dynspread(
                    img,
                    threshold=0.5,
                    max_px=max_px,
                    shape='circle',
                ).to_pil()

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
        marker = {}

    # Build map figure
    map_graph = {
        'data': [{
            'type': 'scattermapbox',
            'lat': lat, 'lon': lon,
            'customdata': customdata,
            'marker': marker,
            'hovertemplate': (
                "sex: %{customdata[0]}<br>"
                "<extra></extra>"
            )
        }],
        'layout': {
            'template': template,
            'uirevision': True,
            'mapbox': {
                'style': "dark",
                'accesstoken': token,
                'layers': layers,
            },
            'margin': {"r": 0, "t": 0, "l": 0, "b": 0},
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
            }]
        },
    }

    map_graph['layout']['mapbox'].update(position)

    return map_graph

def scatter_bubble_2d(df, columnx, columny, selections, query_cache, colorscale_name, colorscale_transform, aggregate, aggregate_column):
    """
    Build histogram figure

    Args:
        df: pandas or cudf DataFrame
        columnx: Column name for x axis
        columny: Column name for y axis
        selections: Dictionary from column names to query expressions
        query_cache: Dict from query expression to filtered DataFrames
        colorscale_name: Name of plotly colorscale
        colorscale_transform: Colorscale transformation

    Returns:
        2d scatter plot
    """
    query = build_query(selections, columnx)
    if query in query_cache:
        df = query_cache[query]
    elif query:
        df = df.query(query)
        query_cache[query] = df

    if isinstance(df, cudf.DataFrame):
        temp = df.groupby([columny,columnx])['sex'].count().to_pandas().reset_index(level=[0,1])
    else:
        temp = df.groupby([columny,columnx])['sex'].count().reset_index(level=[0,1])
    x = temp[columnx].values
    y = temp[columny].values
    size = temp['sex'].values
    color_temp = {'color': size}
    colorscale = build_colorscale(colorscale_name, colorscale_transform, aggregate, aggregate_column)

    if aggregate == 'count_cat' and aggregate_column == columnx:
        color_temp['color'] = x
        colorscale = [(v, clr) for v, clr in zip(mappings[columnx].keys(), colors[columnx])]
    if aggregate == 'count_cat' and aggregate_column == columny:
        color_temp['color'] = y
        colorscale = [(v, clr) for v, clr in zip(mappings[columny].keys(), colors[columny])]

    
    fig = {
        'data': [{
            'type': 'scatter', 'x': x, 'y': y,
            'mode':'markers',
            'marker': dict(size=np.log2(size),
                **color_temp,
                colorscale=colorscale,colorbar=dict(
                    title='Number of people',
                    thickness=20
                    )),
            'text':size,
            'hovertemplate': "<b> %{y}</b> </br> </br>" +
                "%{xaxis.title.text}: %{x}<br>" +
                "Number of people: %{text:,}"
        },
        ]
    }

    if columnx not in selections:
        fig['data'][0]['selectedpoints'] = False

    return fig

def build_histogram_default_bins(df, column, selections, query_cache, orientation,colorscale_name, colorscale_transform, aggregate, aggregate_column):
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

    color_scale = build_colorscale(colorscale_name, colorscale_transform, aggregate, aggregate_column)
    marker = {'color': text_color}
    if aggregate == 'count_cat' and column == aggregate_column:
        colorscale = [clr for v, clr in zip(mappings[aggregate_column].keys(), colors[aggregate_column])]
        marker = {'color': colorscale}

    range_ages = [20,40,60,84, 85]
    colors_ages = ['#C700E5','#9D00DB', '#7300D2','#4900C8','#1F00BF']
    labels = ['0-20', '21-40', '41-60', '60-84', '85+']
    # centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    if orientation == 'h':
        fig = {
            'data':[],
            'layout': {
                'xaxis': {
                    'type': 'log',
                    'range': [0, 8],  # Up to 100M
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
                    'type': 'log',
                    'range': [0, 8],  # Up to 100M
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
                'name':labels[index]
                }
            )
        else:
            fig['data'].append(
                {
                'type': 'bar', 'x': bin_edges_temp, 'y': count_temp,
                'marker': {'color': colors_ages[index]},
                'name':labels[index]
                }
            )
        
    if column not in selections:
        for i in range(len(fig['data'])):
            fig['data'][i]['selectedpoints'] = False

    return fig


def build_updated_figures(
        df, relayout_data, selected_age, selected_scatter_graph,
        aggregate, colorscale_name, selected_cow, selected_sex
):
    """
    Build all figures for dashboard

    Args:
        df: pandas or cudf DataFrame
        relayout_data: relayout_data for datashader figure
        selected_age_male: selectedData for age-male histogram
        selected_age_female: selectedData for age-female histogram
        selected_cow: selectedData for class of worker histogram
        selected_scatter_graph: selectedData for education-income scatter plot
        aggregate: Aggregate operation for choropleth (count, mean, etc.)
        aggregate_column: Aggregate column for choropleth
        colorscale_name: Colorscale name from plotly.colors.sequential
        colorscale_transform: Colorscale transformation ('linear', 'sqrt', 'cbrt', 'log')

    Returns:
        tuple of figures in the following order
        (relayout_data, age_male_histogram, age_female_histogram,
        cow_histogram, scatter_graph,
        n_selected_indicator)
    """
    global data_3857, data_center_3857, data_4326, data_center_4326

    colorscale_transform, aggregate_column = 'linear', 'sex'
    selected = {}

    if selected_age:
        selected = {
            'age': bar_selection_to_query(selected_age, 'age')
        }

    array_module = cupy if isinstance(df, cudf.DataFrame) else np
    all_hists_query = build_query(selected)

    drop_down_queries = ''
    if selected_sex != -1:
        drop_down_queries = 'sex == @selected_sex'

    if selected_cow != -1:
        if len(drop_down_queries) == 0:
            drop_down_queries = 'cow == @selected_cow'
        else:
            drop_down_queries += ' and cow == @selected_cow'
    
    if len(drop_down_queries) > 0:
        df = df.query(drop_down_queries)

    isin_mask_scatter = None

    if selected_scatter_graph:
        selected_pincp = array_module.array([p['x'] for p in selected_scatter_graph['points']])
        selected_schl = array_module.array([p['y'] for p in selected_scatter_graph['points']])
        pincp_array = df.income.values
        schl_array = df.education.values
        isin_mask1 = array_module.zeros(len(pincp_array), dtype=np.bool)
        isin_mask2 = array_module.zeros(len(schl_array), dtype=np.bool)
        
        stride = 32
        for i in range(0, len(selected_pincp), stride):
            zips_chunk = selected_pincp[i:i+stride]
            isin_mask1 |= array_module.isin(pincp_array, zips_chunk)
        for i in range(0, len(selected_schl), stride):
            zips_chunk = selected_schl[i:i+stride]
            isin_mask2 |= array_module.isin(schl_array, zips_chunk)

        isin_mask_scatter = array_module.logical_and(isin_mask1, isin_mask2)
        df_scatter = df[isin_mask_scatter]
    else:
        selected_pincp = None
        selected_schl = None
        df_scatter = df
    
    # if relayout_data is not None:
    transformer_4326_to_3857 = Transformer.from_crs("epsg:4326", "epsg:3857")
    def epsg_4326_to_3857(coords):
        return [transformer_4326_to_3857.transform(*reversed(row)) for row in coords]
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
    query_expr_xy = f"(x >= {x0}) & (x <= {x1}) & (y >= {y0}) & (y <= {y1})"
    df_map = df_scatter.query(query_expr_xy)

    datashader_plot = build_datashader_plot(
        df_scatter.query(all_hists_query) if all_hists_query else df_scatter, aggregate,
        aggregate_column, colorscale_name, colorscale_transform, new_coordinates, position, x_range, y_range)

    df_hists = df_map.query(all_hists_query) if all_hists_query else df_map
    # Build indicator figure
    n_selected_indicator = {
        'data': [{
            'type': 'indicator',
            'value': len(
                df_hists
            ),
            'number': {
                'font': {
                    'color': text_color
                },
                "valueformat": ","
            }
        }],
        'layout': {
            'template': template,
            'height': row_heights[0],
            'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10}
        }
    }
    
    query_cache = {}


    if isinstance(df_map, cudf.DataFrame):
        df_map = df_map.groupby('age')['x'].count().to_pandas()
    else:
        df_map = df_map.groupby('age')['x'].count()

    age_histogram = build_histogram_default_bins(
        df_map, 'age', selected, query_cache,'v', colorscale_name, colorscale_transform, aggregate, aggregate_column
    )
    
    
    scatter_graph = scatter_bubble_2d(
        df_hists,'income', 'education', selected, query_cache, colorscale_name, colorscale_transform, aggregate, aggregate_column
    )

    scatter_graph['layout'] = {
            'xaxis': {
                'title': {
                    'text': "Income ($)"
                },
                'ticktext': list(mappings['income'].values()),
                'tickvals': list(mappings['income'].keys())
            },
            'yaxis': {
                'title': {
                    'text': "Education Category"
                },
                'ticktext': list(mappings['education'].values()),
                'tickvals': list(mappings['education'].keys())
            },
            'hovermode': 'closest',
            'dragmode': 'select',
            'template': template,
            'uirevision': True,
        }

    return (datashader_plot, age_histogram, scatter_graph,
        n_selected_indicator,)


def register_update_plots_callback(client):
    """
    Register Dash callback that updates all plots in response to selection events
    Args:
        df_d: Dask.delayed pandas or cudf DataFrame
    """
    @app.callback(
        [Output('indicator-graph', 'figure'), Output('map-graph', 'figure'),
         Output('age-histogram', 'figure'), Output('scatter-graph', 'figure')
         ],
        [Input('map-graph', 'relayoutData'), Input('age-histogram', 'selectedData'),
            Input('scatter-graph', 'selectedData'),
            Input('aggregate-dropdown', 'value'), Input('colorscale-dropdown', 'value'),
            Input('cow-dropdown', 'value'), Input('sex-dropdown', 'value'),
            Input('gpu-toggle', 'on')
        ]
    )
    def update_plots(
            relayout_data, selected_age, selected_scatter_graph,
            aggregate, colorscale_name, selected_cow, selected_sex, gpu_enabled
    ):
        global data_3857, data_center_3857, data_4326, data_center_4326

        t0 = time.time()

    
        # Get delayed dataset from client
        if gpu_enabled:
            df_d = client.get_dataset('c_df_d')
        else:
            df_d = client.get_dataset('pd_df_d')

        if data_3857 == []:
            projections = delayed(set_projection_bounds)(df_d)
            data_3857, data_center_3857, data_4326, data_center_4326 = projections.compute()

        figures_d = delayed(build_updated_figures)(
            df_d, relayout_data, selected_age, selected_scatter_graph,
            aggregate, colorscale_name, selected_cow, selected_sex
        )

        figures = figures_d.compute()

        (datashader_plot, age_histogram, scatter_graph,
        n_selected_indicator,) = figures

        print(f"Update time: {time.time() - t0}")
        return (
            n_selected_indicator, datashader_plot, age_histogram, scatter_graph
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


def publish_dataset_to_cluster():

    census_data_url = 'https://s3.us-east-2.amazonaws.com/rapidsai-data/viz-data/census_data.parquet.tar.gz'
    data_path = "../data/census_data.parquet/*"
    check_dataset(census_data_url, data_path)

    # Note: The creation of a Dask LocalCluster must happen inside the `__main__` block,
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0")
    client = Client(cluster)
    print(f"Dask status: {cluster.dashboard_link}")

    # Load dataset and persist dataset on cluster
    def load_and_publish_dataset():
        # cudf DataFrame
        c_df_d = delayed(load_dataset)(data_path).persist()
        # pandas DataFrame
        pd_df_d = delayed(c_df_d.to_pandas)().persist()

        # print(type(c_df_d))
        # Unpublish datasets if present
        for ds_name in ['pd_df_d', 'c_df_d']:
            if ds_name in client.datasets:
                client.unpublish_dataset(ds_name)

        # Publish datasets to the cluster
        client.publish_dataset(pd_df_d=pd_df_d)
        client.publish_dataset(c_df_d=c_df_d)

    load_and_publish_dataset()

    # Precompute field bounds
    c_df_d = client.get_dataset('c_df_d')
    
    # Define callback to restart cluster and reload datasets
    @app.callback(
        Output('reset-gpu-complete', 'children'),
        [Input('reset-gpu', 'n_clicks')]
    )
    def restart_cluster(n_clicks):
        if n_clicks:
            print("Restarting LocalCUDACluster")
            client.unpublish_dataset('pd_df_d')
            client.unpublish_dataset('c_df_d')
            client.restart()
            load_and_publish_dataset()

    # Register top-level callback that updates plots
    register_update_plots_callback(client)


def server():
    # gunicorn entry point when called with `gunicorn 'app:server()'`
    publish_dataset_to_cluster()
    return app.server


if __name__ == '__main__':
    # development entry point
    publish_dataset_to_cluster()

    # Launch dashboard
    app.run_server(debug=False, dev_tools_silence_routes_logging=True, host='0.0.0.0')