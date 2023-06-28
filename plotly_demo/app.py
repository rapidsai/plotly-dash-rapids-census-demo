import os
import time

import cudf
import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np
import pandas as pd
from dash import Dash, ctx, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from distributed import Client
from utils import *
import tarfile

# ### Dashboards start here
text_color = "#cfd8dc"  # Material blue-grey 100

DATA_PATH = "../data"
DATA_PATH_STATE = f"{DATA_PATH}/state-wise-population"
DATA_PATH_TOTAL = f"{DATA_PATH}/total_population_dataset.parquet"

# Download the required states data
census_data_url = "https://data.rapids.ai/viz-data/total_population_dataset.parquet"
check_dataset(census_data_url, DATA_PATH_TOTAL)

census_state_data_url = "https://data.rapids.ai/viz-data/state-wise-population.tar.xz"
if not os.path.exists(DATA_PATH_STATE):
    check_dataset(census_state_data_url, f"{DATA_PATH_STATE}.tar.xz")
    print("Extracting state-wise-population.tar.xz ...")
    with tarfile.open(f"{DATA_PATH_STATE}.tar.xz", "r:xz") as tar:
        tar.extractall(DATA_PATH)
    print("Done.")

state_files = os.listdir(DATA_PATH_STATE)
state_names = [os.path.splitext(f)[0] for f in state_files]
# add USA(combined dataset) to the list of states
state_names.append("USA")


(
    data_center_3857,
    data_3857,
    data_4326,
    data_center_4326,
    selected_map_backup,
    selected_race_backup,
    selected_county_top_backup,
    selected_county_bt_backup,
    view_name_backup,
    c_df,
    gpu_enabled_backup,
    dragmode_backup,
    currently_loaded_state,
) = ([], [], [], [], None, None, None, None, None, None, None, "pan", None)


app = Dash(__name__)
application = app.server

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children=[
                        "Census 2020 Net Migration Visualization",
                        html.A(
                            html.Img(
                                src="assets/rapids-logo.png",
                                style={
                                    "float": "right",
                                    "height": "45px",
                                    "marginRight": "1%",
                                    "marginTop": "-7px",
                                },
                            ),
                            href="https://rapids.ai/",
                        ),
                        html.A(
                            html.Img(
                                src="assets/dash-logo.png",
                                style={"float": "right", "height": "30px"},
                            ),
                            href="https://dash.plot.ly/",
                        ),
                    ],
                    style={"textAlign": "left"},
                ),
            ]
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.H4(
                                    [
                                        "Population Count and Query Time",
                                    ],
                                    className="container_title",
                                ),
                                dcc.Loading(
                                    dcc.Graph(
                                        id="indicator-graph",
                                        figure=blank_fig(row_heights[3]),
                                        config={"displayModeBar": False},
                                    ),
                                    color="#b0bec5",
                                    style={"height": f"{row_heights[3]}px"},
                                ),
                            ],
                            style={"height": f"{row_heights[0]}px"},
                            className="five columns pretty_container",
                            id="indicator-div",
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Button(
                                            "Clear All Selections",
                                            id="clear-all",
                                            className="reset-button",
                                        ),
                                    ]
                                ),
                                html.H4(
                                    [
                                        "Options",
                                    ],
                                    className="container_title",
                                ),
                                html.Table(
                                    [
                                        html.Tr(
                                            [
                                                html.Td(
                                                    html.Div("GPU Acceleration"),
                                                    className="config-label",
                                                ),
                                                html.Td(
                                                    html.Div(
                                                        [
                                                            daq.DarkThemeProvider(
                                                                daq.BooleanSwitch(
                                                                    on=True,
                                                                    color="#00cc96",
                                                                    id="gpu-toggle",
                                                                )
                                                            ),
                                                            dbc.Tooltip(
                                                                "Caution: Using CPU compute for more than 50 million points is not recommended.",
                                                                target="gpu-toggle",
                                                                placement="bottom",
                                                                autohide=True,
                                                                style={
                                                                    "textAlign": "left",
                                                                    "fontSize": "15px",
                                                                    "color": "white",
                                                                    "width": "350px",
                                                                    "padding": "15px",
                                                                    "borderRadius": "5px",
                                                                    "backgroundColor": "#2a2a2e",
                                                                },
                                                            ),
                                                        ]
                                                    )
                                                ),
                                                #######  State Selection Dropdown ######
                                                html.Td(
                                                    html.Div("Select State"),
                                                    style={"fontSize": "20px"},
                                                ),
                                                html.Td(
                                                    dcc.Dropdown(
                                                        id="state-dropdown",
                                                        options=[
                                                            {"label": i, "value": i}
                                                            for i in state_names
                                                        ],
                                                        value="USA",
                                                    ),
                                                    style={
                                                        "width": "25%",
                                                        "height": "15px",
                                                    },
                                                ),
                                                ###### VIEWS ARE HERE ###########
                                                html.Td(
                                                    html.Div("Data-Selection"),
                                                    style={"fontSize": "20px"},
                                                ),  # className="config-label"
                                                html.Td(
                                                    dcc.Dropdown(
                                                        id="view-dropdown",
                                                        options=[
                                                            {
                                                                "label": "Total Population",
                                                                "value": "total",
                                                            },
                                                            {
                                                                "label": "Migrating In",
                                                                "value": "in",
                                                            },
                                                            {
                                                                "label": "Stationary",
                                                                "value": "stationary",
                                                            },
                                                            {
                                                                "label": "Migrating Out",
                                                                "value": "out",
                                                            },
                                                            {
                                                                "label": "Net Migration",
                                                                "value": "net",
                                                            },
                                                            {
                                                                "label": "Population with Race",
                                                                "value": "race",
                                                            },
                                                        ],
                                                        value="in",
                                                        searchable=False,
                                                        clearable=False,
                                                    ),
                                                    style={
                                                        "width": "25%",
                                                        "height": "15px",
                                                    },
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={"width": "100%", "marginTop": "30px"},
                                ),
                                # Hidden div inside the app that stores the intermediate value
                                html.Div(
                                    id="datapoints-state-value",
                                    style={"display": "none"},
                                ),
                            ],
                            style={"height": f"{row_heights[0]}px"},
                            className="seven columns pretty_container",
                            id="config-div",
                        ),
                    ]
                ),
                ##################### Map starts  ###################################
                html.Div(
                    children=[
                        html.Button(
                            "Clear Selection", id="reset-map", className="reset-button"
                        ),
                        html.H4(
                            [
                                "Population Distribution of Individuals",
                            ],
                            className="container_title",
                        ),
                        dcc.Graph(
                            id="map-graph",
                            config={"displayModeBar": False},
                            figure=blank_fig(row_heights[1]),
                        ),
                        # Hidden div inside the app that stores the intermediate value
                        html.Div(
                            id="intermediate-state-value", style={"display": "none"}
                        ),
                    ],
                    className="twelve columns pretty_container",
                    id="map-div",
                    style={"height": "50%"},
                ),
                ################# Bars start #########################
                # Race start
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Button(
                                    "Clear Selection",
                                    id="clear-race",
                                    className="reset-button",
                                ),
                                html.H4(
                                    [
                                        "Race Distribution",
                                    ],
                                    className="container_title",
                                ),
                                dcc.Graph(
                                    id="race-histogram",
                                    config={"displayModeBar": False},
                                    figure=blank_fig(row_heights[2]),
                                ),
                            ],
                            className="one-third column pretty_container",
                            id="race-div",
                        ),  # County top starts
                        html.Div(
                            children=[
                                html.Button(
                                    "Clear Selection",
                                    id="clear-county-top",
                                    className="reset-button",
                                ),
                                html.H4(
                                    [
                                        "County-wise Top 15",
                                    ],
                                    className="container_title",
                                ),
                                dcc.Graph(
                                    id="county-histogram-top",
                                    config={"displayModeBar": False},
                                    figure=blank_fig(row_heights[2]),
                                    animate=False,
                                ),
                            ],
                            className=" one-third column pretty_container",
                            id="county-div-top",
                        ),
                        # County bottom starts
                        html.Div(
                            children=[
                                html.Button(
                                    "Clear Selection",
                                    id="clear-county-bottom",
                                    className="reset-button",
                                ),
                                html.H4(
                                    [
                                        "County-wise Bottom 15",
                                    ],
                                    className="container_title",
                                ),
                                dcc.Graph(
                                    id="county-histogram-bottom",
                                    config={"displayModeBar": False},
                                    figure=blank_fig(row_heights[2]),
                                    animate=False,
                                ),
                            ],
                            className="one-third column pretty_container",
                        ),
                    ],
                    className="twelve columns",
                )
                ############## End of  Bars #####################
            ]
        ),
        html.Div(
            [
                html.H4("Acknowledgements and Data Sources", style={"marginTop": "0"}),
                dcc.Markdown(
                    """\
- 2020 Population Census and 2010 Population Census to compute Migration Dataset, used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](https://www.nhgis.org/) ( not for redistribution ).
- Base map layer provided by [Mapbox](https://www.mapbox.com/).
- Dashboard developed with [Plotly Dash](https://plotly.com/dash/).
- Geospatial point rendering developed with [Datashader](https://datashader.org/).
- GPU toggle accelerated with [RAPIDS cudf and dask_cudf](https://rapids.ai/) and [cupy](https://cupy.chainer.org/), CPU toggle with [pandas](https://pandas.pydata.org/).
- For source code and data workflow, visit our [GitHub](https://github.com/rapidsai/plotly-dash-rapids-census-demo/tree/master).
"""
                ),
            ],
            style={
                "width": "98%",
                "marginRight": "0",
                "padding": "10px",
            },
            className="twelve columns pretty_container",
        ),
    ],
)


# Clear/reset button callbacks
@app.callback(
    Output("map-graph", "selectedData"),
    [Input("reset-map", "n_clicks"), Input("clear-all", "n_clicks")],
)
def clear_map(*args):
    return None


@app.callback(
    Output("race-histogram", "selectedData"),
    [Input("clear-race", "n_clicks"), Input("clear-all", "n_clicks")],
)
def clear_race_hist_selections(*args):
    return None


@app.callback(
    Output("county-histogram-top", "selectedData"),
    [Input("clear-county-top", "n_clicks"), Input("clear-all", "n_clicks")],
)
def clear_county_hist_top_selections(*args):
    return None


@app.callback(
    Output("county-histogram-bottom", "selectedData"),
    [Input("clear-county-bottom", "n_clicks"), Input("clear-all", "n_clicks")],
)
def clear_county_hist_bottom_selections(*args):
    return None


@app.callback(
    [
        Output("indicator-graph", "figure"),
        Output("map-graph", "figure"),
        Output("map-graph", "config"),
        Output("county-histogram-top", "figure"),
        Output("county-histogram-top", "config"),
        Output("county-histogram-bottom", "figure"),
        Output("county-histogram-bottom", "config"),
        Output("race-histogram", "figure"),
        Output("race-histogram", "config"),
        Output("intermediate-state-value", "children"),
    ],
    [
        Input("map-graph", "relayoutData"),
        Input("map-graph", "selectedData"),
        Input("race-histogram", "selectedData"),
        Input("county-histogram-top", "selectedData"),
        Input("county-histogram-bottom", "selectedData"),
        Input("view-dropdown", "value"),
        Input("state-dropdown", "value"),
        Input("gpu-toggle", "on"),
    ],
    [
        State("intermediate-state-value", "children"),
    ],
)
def update_plots(
    relayout_data,
    selected_map,
    selected_race,
    selected_county_top,
    selected_county_bottom,
    view_name,
    state_name,
    gpu_enabled,
    coordinates_backup,
):
    global data_3857, data_center_3857, data_4326, data_center_4326, currently_loaded_state, selected_race_backup, selected_county_top_backup, selected_county_bt_backup

    # condition to avoid reloading on tool update
    if (
        ctx.triggered_id == "map-graph"
        and relayout_data
        and list(relayout_data.keys()) == ["dragmode"]
    ):
        raise PreventUpdate

    # condition to avoid a bug in plotly where selectedData is reset following a box-select
    if not (selected_race is not None and len(selected_race["points"]) == 0):
        selected_race_backup = selected_race
    elif ctx.triggered_id == "race-histogram":
        raise PreventUpdate

    # condition to avoid a bug in plotly where selectedData is reset following a box-select
    if not (
        selected_county_top is not None and len(selected_county_top["points"]) == 0
    ):
        selected_county_top_backup = selected_county_top
    elif ctx.triggered_id == "county-histogram-top":
        raise PreventUpdate

    # condition to avoid a bug in plotly where selectedData is reset following a box-select
    if not (
        selected_county_bottom is not None
        and len(selected_county_bottom["points"]) == 0
    ):
        selected_county_bt_backup = selected_county_bottom
    elif ctx.triggered_id == "county-histogram-bottom":
        raise PreventUpdate

    df = read_dataset(state_name, gpu_enabled, currently_loaded_state)

    t0 = time.time()

    if coordinates_backup is not None:
        coordinates_4326_backup, position_backup = coordinates_backup
    else:
        coordinates_4326_backup, position_backup = None, None

    colorscale_name = "Viridis"

    if data_3857 == [] or state_name != currently_loaded_state:
        (
            data_3857,
            data_center_3857,
            data_4326,
            data_center_4326,
        ) = set_projection_bounds(df)

    (
        datashader_plot,
        race_histogram,
        county_top_histogram,
        county_bottom_histogram,
        n_selected_indicator,
        coordinates_4326_backup,
        position_backup,
    ) = build_updated_figures(
        df,
        relayout_data,
        selected_map,
        selected_race_backup,
        selected_county_top_backup,
        selected_county_bt_backup,
        colorscale_name,
        data_3857,
        data_center_3857,
        data_4326,
        data_center_4326,
        coordinates_4326_backup,
        position_backup,
        view_name,
    )

    barchart_config = {
        "displayModeBar": True,
        "modeBarButtonsToRemove": [
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "resetScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
        ],
    }
    compute_time = time.time() - t0
    print(f"Query time: {compute_time}")
    n_selected_indicator["data"].append(
        {
            "title": {"text": "Query Time"},
            "type": "indicator",
            "value": round(compute_time, 4),
            "domain": {"x": [0.6, 0.85], "y": [0, 0.5]},
            "number": {
                "font": {
                    "color": text_color,
                    "size": "50px",
                },
                "suffix": " seconds",
            },
        }
    )

    datashader_plot["layout"]["dragmode"] = (
        relayout_data["dragmode"]
        if (relayout_data and "dragmode" in relayout_data)
        else dragmode_backup
    )
    # update currently loaded state
    currently_loaded_state = state_name

    return (
        n_selected_indicator,
        datashader_plot,
        {
            "displayModeBar": True,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "zoomInMapbox",
                "zoomOutMapbox",
                "toggleHover",
            ],
        },
        county_top_histogram,
        barchart_config,
        county_bottom_histogram,
        barchart_config,
        race_histogram,
        barchart_config,
        (coordinates_4326_backup, position_backup),
    )


def read_dataset(state_name, gpu_enabled, currently_loaded_state):
    global c_df
    if state_name != currently_loaded_state:
        if state_name == "USA":
            data_path = f"{DATA_PATH}/total_population_dataset.parquet"
        else:
            data_path = f"{DATA_PATH_STATE}/{state_name}.parquet"
        c_df = load_dataset(data_path, "cudf" if gpu_enabled else "pandas")
    return c_df


if __name__ == "__main__":
    # Launch dashboard
    app.run_server(
        debug=True,
        dev_tools_hot_reload=True,
        host="0.0.0.0",
    )
