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
from dash.dependencies import Input, Output, State
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
row_heights = [150, 440, 200, 540]
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
mappings_hover = {}
# Load mapbox token from environment variable or file
token = os.getenv('MAPBOX_TOKEN')
if not token:
    token = open(".mapbox_token").read()

mappings_hover['cow'] = {
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
    0: "Emp",
    1: "Self-emp",
    2: "Emp non-profit",
    3: "Local gov emp",
    4: "State gov emp",
    5: "Federal gov emp",
    6: "Self-emp non-business",
    7: "Unpaid workers",
    8: "below age 16",
}

mappings['sex'] = {
    -1: "All genders",
    0: 'Males',
    1: 'Females'
}

mappings_hover['education']= {
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


mappings['education'] = {
    0: "No school",
    1: "Upto 4th",
    2: "5th & 6th",
    3: "7th & 8th",
    4: "9th",
    5: "10th",
    6: "11th",
    7: "12th",
    8: "High school",
    9: "College(<1 yr)",
    10: "College(no degree)",
    11: "Associate's",
    12: "Bachelor's",
    13: "Master's",
    14: "Prof. school",
    15: "Doctorate",
    16: 'below age 16'
}


mappings_hover['income'] = {
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

mappings['income'] = {
    0: "$2,499",
    1: "$4,999",
    2: "$7,499",
    3: "$9,999",
    4: "$12,499",
    5: "$14,999",
    6: "$17,499",
    7: "$19,999",
    8: "$22,499",
    9: "$24,999",
    10: "$29,999",
    11: "$34,999",
    12: "$39,999",
    13: "$44,999",
    14: "$49,999",
    15: "$54,999",
    16: "$64,999",
    17: "$74,999",
    18: "$99,999",
    19: "$100,000+",
    20: 'below age 25',
}


data_center_3857, data_3857, data_4326, data_center_4326 = [], [], [], []


def load_dataset(path):
    """
    Args:
        path: Path to arrow file containing mortgage dataset

    Returns:
        pandas DataFrame
    """
    if os.path.isdir(path):
        path = path + '/*'
    df_d = cudf.read_parquet(path)
    df_d['sex'] = df_d.sex.to_pandas().astype('category')
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
            'Census 2010 Visualization',
            html.A(
                html.Img(
                    src="assets/rapids-logo.png",
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
                html.H4([
                    "Population Count",
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
                html.Div(children=[
                    html.Button(
                        "Clear All Selections", id='clear-all', className='reset-button'
                    ),
                ]),
                html.H4([
                    "Options",
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
                                for cs in ['Viridis', 'Cividis', 'Magma']
                            ],
                            value='Viridis',
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
                "Population Distribution of Individuals",
            ], className="container_title"),
            dcc.Graph(
                id='map-graph',
                figure=blank_fig(row_heights[1]),
            ),
            # Hidden div inside the app that stores the intermediate value
            html.Div(id='intermediate-state-value', style={'display': 'none'})

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
                    html.Button(
                        "Clear Selection", id='clear-education', className='reset-button'
                    ),
                    html.H4([
                        "Education Distribution",
                    ], className="container_title"),

                    dcc.Graph(
                        id='education-histogram',
                        config={'displayModeBar': False},
                        figure=blank_fig(row_heights[1]),
                        animate=True
                    ),
                ],
                className='twelve columns pretty_container', id="education-div"
            )
        ]),
        html.Div(children=[
            html.Div(
                children=[
                    html.Button(
                        "Clear Selection", id='clear-income', className='reset-button'
                    ),
                    html.H4([
                        "Income Distribution",
                    ], className="container_title"),

                    dcc.Graph(
                        id='income-histogram',
                        config={'displayModeBar': False},
                        figure=blank_fig(row_heights[1]),
                        animate=True
                    ),
                ],
                className='twelve columns pretty_container', id="income-div"
            )
        ]),
        html.Div(children=[
            html.Div(
                children=[
                    html.Button(
                        "Clear Selection", id='clear-cow', className='reset-button'
                    ),
                    html.H4([
                        "Class of Workers Distribution",
                    ], className="container_title"),

                    dcc.Graph(
                        id='cow-histogram',
                        config={'displayModeBar': False},
                        figure=blank_fig(row_heights[1]),
                        animate=True
                    ),
                ],
                className='twelve columns pretty_container', id="cow-div"
            )
        ]),
        html.Div(children=[
            html.Div(
                children=[
                    html.Button(
                        "Clear Selection", id='clear-age', className='reset-button'
                    ),
                    html.H4([
                        "Age Distribution",
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
            html.H4('Acknowledgements and Data Sources', style={"margin-top": "0"}),
            dcc.Markdown('''\
- 2010 Population Census and 2018 ACS data used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](https://www.nhgis.org/) ( not for redistribution )
- Base map layer provided by mapbox
- Dashboard developed with Plot.ly Dash
- Geospatial point rendering developed with Datashader
- GPU accelerated with RAPIDS cudf and cupy libraries. CPU using pandas libraries.
- For source code visit our GitHub
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

# Clear/reset button callbacks
@app.callback(
    Output('map-graph', 'selectedData'),
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
    Output('education-histogram', 'selectedData'),
    [Input('clear-education', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_education_hist_selections(*args):
    return None

@app.callback(
    Output('income-histogram', 'selectedData'),
    [Input('clear-income', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_income_hist_selections(*args):
    return None

@app.callback(
    Output('cow-histogram', 'selectedData'),
    [Input('clear-cow', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_cow_hist_selections(*args):
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
def build_colorscale(colorscale_name, transform):
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
        datashader_color_scale['cmap'] = [i[1] for i in build_colorscale(colorscale_name, colorscale_transform)]
        if not isinstance(df, cudf.DataFrame):
            df[aggregate_column] = df[aggregate_column].astype('int8')

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
    else:
        # Shade aggregation into an image that we can add to the map as a mapbox
        # image layer
        img = tf.shade(agg, **datashader_color_scale).to_pil()
    
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

def build_histogram_default_bins(
    df, column, selections, query_cache,
    orientation, colorscale_name, colorscale_transform
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
    query = build_query(selections, column)
    if query in query_cache:
        df = query_cache[query]
    elif query:
        df = df.query(query)
        query_cache[query] = df

    df = df.groupby(column)['x'].count().to_pandas()

    bin_edges = df.index.values
    counts = df.values

    mapping_options = {}
    xaxis_labels = {}
    if column in mappings:
        if column in mappings_hover:
            mapping_options = {
            'text': list(mappings_hover[column].values()),
            'hovertemplate': "%{text}: %{y} <extra></extra>"
            }
        else:
            mapping_options = {
                'text': list(mappings[column].values()),
                'hovertemplate': "%{text} : %{y} <extra></extra>"
            }
        xaxis_labels = {
            'tickvals': list(mappings[column].keys()),
            'ticktext': list(mappings[column].values())
        }
    
    # color_scale = build_colorscale(colorscale_name, colorscale_transform)

    # centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    if orientation == 'h':
        fig = {
            'data':[{
            'type': 'bar', 'x': bin_edges, 'y': counts,
            'marker': {
                'color': counts,
                'colorscale': colorscale_name
            },
            **mapping_options
        }],
            'layout': {
                'xaxis': {
                    'type': 'linear',
                    'title': {
                        'text': "Count"
                    },
                },
                'yaxis': {
                    **xaxis_labels
                },
                'selectdirection': 'v',
                'dragmode': 'select',
                'template': template,
                'uirevision': True,
                'hovermode': 'closest'
            }
        }
    else:
        fig = {
            'data': [{
                'type': 'bar', 'x': bin_edges, 'y': counts,
                'marker': {
                    'color': counts,
                    'colorscale': colorscale_name
                },
                **mapping_options

            }],
            'layout': {
                'yaxis': {
                    'type': 'linear',
                    'title': {
                        'text': "Count"
                    },
                },
                'xaxis': {
                    **xaxis_labels
                },
                'selectdirection': 'h',
                'dragmode': 'select',
                'template': template,
                'uirevision': True,
                'hovermode': 'closest'
            }
        }
    
    if column not in selections:
        for i in range(len(fig['data'])):
            fig['data'][i]['selectedpoints'] = False

    return fig


def build_updated_figures(
        df, relayout_data, selected_map, selected_education,
        selected_income, selected_cow, selected_age, aggregate,
        colorscale_name, data_3857, data_center_3857, data_4326,
        data_center_4326, coordinates_4326_backup, position_backup
):
    """
    Build all figures for dashboard

    Args:
        - df: census 2010 dataset (cudf.DataFrame)
        - df_hospitals: hospitals dataset (cudf.DataFrame)
        - df_covid: covid dataset (cudf.DataFrame)
        - relayout_data: plotly relayout object(dict) for datashader figure
        - selected_map: selected_map dictionary object from plotly box-select
        - aggregate: aggregate function (str)
        - aggregate_column: aggregate column (census column name)
        - population_enabled: Bool
        - population_colorscale: colorscale name (str)
        - hospital_enabled: Bool
        - hospital_colorscale: colorscale name (str)
        - covid_enabled: Bool
        - covid_count_type: count/count_cat
        - data_3857
        - data_center_3857
        - data_4326
        - data_center_4326
        - coordinates_4326_backup
        - position_backup

    Returns:
        tuple of figures in the following order
        (relayout_data, age_male_histogram, age_female_histogram,
        cow_histogram, scatter_graph,
        n_selected_indicator)
    """
    colorscale_transform, aggregate_column = 'linear', 'sex'
    selected = {}

    selected = {
        col: bar_selection_to_query(sel, col)
        for col, sel in zip([
            'education', 'income',
            'cow', 'age'
        ], [
            selected_education, selected_income, selected_cow, selected_age
        ]) if sel and sel.get('points', [])
    }

    array_module = cupy if isinstance(df, cudf.DataFrame) else np
    
    all_hists_query = build_query(selected)

    # if relayout_data is not None:
    transformer_4326_to_3857 = Transformer.from_crs("epsg:4326", "epsg:3857")
    def epsg_4326_to_3857(coords):
        return [transformer_4326_to_3857.transform(*reversed(row)) for row in coords]
    
    coordinates_4326 = relayout_data and relayout_data.get('mapbox._derived', {}).get('coordinates', None)
    dragmode = relayout_data and 'dragmode' in relayout_data and coordinates_4326_backup is not None

    if dragmode:
        coordinates_4326 = coordinates_4326_backup
        coordinates_3857 = epsg_4326_to_3857(coordinates_4326)
        position = position_backup
    elif coordinates_4326:
        lons, lats = zip(*coordinates_4326)
        lon0, lon1 = max(min(lons), data_4326[0][0]), min(max(lons), data_4326[1][0])
        lat0, lat1 = max(min(lats), data_4326[0][1]), min(max(lats), data_4326[1][1])
        coordinates_4326 = [
            [lon0, lat0],
            [lon1, lat1],
        ]
        coordinates_3857 = epsg_4326_to_3857(coordinates_4326)
        coordinates_4326_backup = coordinates_4326

        position = {
            'zoom': relayout_data.get('mapbox.zoom', None),
            'center': relayout_data.get('mapbox.center', None)
        }
        position_backup = position

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

    def query_df(df, x0, x1, y0, y1, x, y):
        mask_ = (df[x]>=x0) & (df[x]<=x1) & (df[y]<=y0) & (df[y]>=y1)
        if(mask_.sum() != len(df)):
            df = df[mask_]
            df.index = cudf.core.RangeIndex(0,len(df))
        del(mask_)
        return df
        
    if selected_map is not None:
        coordinates_4326 = selected_map['range']['mapbox']
        coordinates_3857 = epsg_4326_to_3857(coordinates_4326)
        x_range_t, y_range_t = zip(*coordinates_3857)
        x0, x1 = x_range_t
        y0, y1 = y_range_t
        df = query_df(df, x0, x1, y0, y1, 'x', 'y')

    datashader_plot = build_datashader_plot(
        df.query(all_hists_query) if all_hists_query else df, aggregate,
        aggregate_column, colorscale_name, colorscale_transform, new_coordinates, position, x_range, y_range)

    df_hists = df.query(all_hists_query) if all_hists_query else df

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

    education_histogram = build_histogram_default_bins(
        df, 'education', selected, query_cache,'v', colorscale_name, colorscale_transform
    )

    income_histogram = build_histogram_default_bins(
        df, 'income', selected, query_cache,'v', colorscale_name, colorscale_transform
    )

    cow_histogram = build_histogram_default_bins(
        df, 'cow', selected, query_cache,'v', colorscale_name, colorscale_transform
    )

    age_histogram = build_histogram_default_bins(
        df, 'age', selected, query_cache,'v', colorscale_name, colorscale_transform
    )


    return (
        datashader_plot, education_histogram, income_histogram,
        cow_histogram, age_histogram, n_selected_indicator,
        coordinates_4326_backup, position_backup
    )


def register_update_plots_callback(client):
    """
    Register Dash callback that updates all plots in response to selection events
    Args:
        df_d: Dask.delayed pandas or cudf DataFrame
    """
    @app.callback(
        [
            Output('indicator-graph', 'figure'), Output('map-graph', 'figure'),
            Output('education-histogram', 'figure'), Output('income-histogram', 'figure'),
            Output('cow-histogram', 'figure'), Output('age-histogram', 'figure'),
            Output('map-graph', 'config'), Output('intermediate-state-value', 'children')
        ],
        [
            Input('map-graph', 'relayoutData'), Input('map-graph', 'selectedData'),
            Input('education-histogram', 'selectedData'), Input('income-histogram', 'selectedData'),
            Input('cow-histogram', 'selectedData'), Input('age-histogram', 'selectedData'), 
            Input('aggregate-dropdown', 'value'), Input('colorscale-dropdown', 'value'),
            Input('gpu-toggle', 'on')
        ],
        [
            State('intermediate-state-value', 'children')
        ]
    )
    def update_plots(
            relayout_data, selected_map, selected_education,
            selected_income, selected_cow, selected_age,
            aggregate, colorscale_name, gpu_enabled, coordinates_backup
    ):
        global data_3857, data_center_3857, data_4326, data_center_4326

        t0 = time.time()

        if coordinates_backup is not None:
            coordinates_4326_backup, position_backup = coordinates_backup
        else:
            coordinates_4326_backup, position_backup = None, None

        # Get delayed dataset from client
        if gpu_enabled:
            df_d = client.get_dataset('c_df_d')
        else:
            df_d = client.get_dataset('pd_df_d')

        if data_3857 == []:
            projections = delayed(set_projection_bounds)(df_d)
            data_3857, data_center_3857, data_4326, data_center_4326 = projections.compute()

        figures_d = delayed(build_updated_figures)(
            df_d, relayout_data, selected_map, selected_education,
            selected_income, selected_cow, selected_age, aggregate,
            colorscale_name, data_3857, data_center_3857, data_4326,
            data_center_4326, coordinates_4326_backup, position_backup
        )

        figures = figures_d.compute()

        (datashader_plot, education_histogram, income_histogram,
        cow_histogram, age_histogram, n_selected_indicator,
        coordinates_4326_backup, position_backup) = figures

        print(f"Update time: {time.time() - t0}")
        return (
            n_selected_indicator, datashader_plot, education_histogram,
            income_histogram, cow_histogram, age_histogram,
            {
                'displayModeBar':True,
                'modeBarButtonsToRemove': ['lasso2d', 'zoomInMapbox', 'zoomOutMapbox', 'toggleHover']
            }, 
            (coordinates_4326_backup, position_backup)
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
    data_path = "../data/census_data.parquet"
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