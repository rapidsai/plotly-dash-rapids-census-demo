import pandas as pd
import cudf
from dash import dcc, html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc
import time
import dash_daq as daq
import dash
from distributed import Client
from utils import *
import os

# ### Dashboards start here
text_color = "#cfd8dc"  # Material blue-grey 100

DATA_PATH = "../data"
DATA_PATH_STATE = f"{DATA_PATH}/state-wise-population"

# Get list of state names from file names
state_files = os.listdir(DATA_PATH_STATE)
state_names = [os.path.splitext(f)[0] for f in state_files]

# append USA to state_names
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
) = ([], [], [], [], None, None, None, None, None, None, None, "pan")


app = dash.Dash(
    __name__,
    external_stylesheets=["assets/s1.css"],
)
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
                                                        value="CALIFORNIA",
                                                        searchable=False,
                                                        clearable=False,
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
                            animate=False,
                        ),
                    ],
                    className="four columns pretty_container",
                    id="race-div",
                    style={"marginRight": "1%"},
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
                    className="four columns pretty_container",
                    id="county-div-top",
                    style={"marginRight": "1%"},
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
                    className="four columns pretty_container",
                    id="county-div-bottom",
                ),
                ############## End of  Bars #####################
            ]
        ),
        html.Div(
            [
                html.H4("Acknowledgements and Data Sources", style={"marginTop": "0"}),
                dcc.Markdown(
                    """\
**Important Data Caveats:** Geospatially filtered data will show accurate distribution, but due to anonymized, multiple cross filtered distributions will not return meaningful results. See [FAQ](https://github.com/rapidsai/plotly-dash-rapids-census-demo/tree/master#faq-and-known-issues) fore details.
- 2010 Population Census and 2018 ACS data used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](https://www.nhgis.org/) ( not for redistribution ).
- Base map layer provided by [Mapbox](https://www.mapbox.com/).
- Dashboard developed with [Plot.ly Dash](https://plotly.com/dash/).
- Geospatial point rendering developed with [Datashader](https://datashader.org/).
- GPU toggle accelerated with [RAPIDS cudf](https://rapids.ai/) and [cupy](https://cupy.chainer.org/), CPU toggle with [pandas](https://pandas.pydata.org/).
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


# # Query string helpers


def register_update_plots_callback():
    """
    Register Dash callback that updates all plots in response to selection events
    Args:
        df_d: Dask.delayed pandas or cudf DataFrame
    """

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
            State("indicator-graph", "figure"),
            State("map-graph", "figure"),
            State("map-graph", "config"),
            State("county-histogram-top", "figure"),
            State("county-histogram-top", "config"),
            State("county-histogram-bottom", "figure"),
            State("county-histogram-bottom", "config"),
            State("race-histogram", "figure"),
            State("race-histogram", "config"),
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
        *backup_args,
    ):
        global data_3857, data_center_3857, data_4326, data_center_4326, selected_map_backup, selected_race_backup, selected_county_top_backup, selected_county_bt_backup, view_name_backup, c_df, gpu_enabled_backup, dragmode_backup
        # condition to avoid reloading on tool update
        if (
            type(relayout_data) == dict
            and list(relayout_data.keys()) == ["dragmode"]
            and selected_map == selected_map_backup
            and selected_race_backup == selected_race
            and selected_county_top_backup == selected_county_top
            and selected_county_bt_backup == selected_county_bottom
            and view_name_backup == view_name
            and gpu_enabled_backup == gpu_enabled
        ):
            backup_args[1]["layout"]["dragmode"] = relayout_data["dragmode"]
            dragmode_backup = relayout_data["dragmode"]
            return backup_args

        selected_map_backup = selected_map
        selected_race_backup = selected_race
        selected_county_top_backup = selected_county_top
        selected_county_bt_backup = selected_county_bottom
        view_name_backup = view_name
        gpu_enabled_backup = gpu_enabled

        t0 = time.time()

        if coordinates_backup is not None:
            coordinates_4326_backup, position_backup = coordinates_backup
        else:
            coordinates_4326_backup, position_backup = None, None

        if state_name == "USA":
            c_df = load_dataset(f"{DATA_PATH}/total_population_dataset.parquet", "cudf")
        else:
            c_df = load_dataset(f"{DATA_PATH_STATE}/{state_name}.parquet", "cudf")

        # Get dataset from client
        if gpu_enabled:
            df = c_df
        else:
            df = c_df.to_pandas()

        colorscale_name = "Viridis"

        if data_3857 == []:
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
            selected_race,
            selected_county_top,
            selected_county_bottom,
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
                "domain": {"x": [0.53, 0.61], "y": [0, 0.5]},
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
            race_histogram,
            barchart_config,
            county_top_histogram,
            barchart_config,
            county_bottom_histogram,
            barchart_config,
            (coordinates_4326_backup, position_backup),
        )


def read_dataset():
    # global c_df
    # census_data_url = "https://rapidsai-data.s3.us-east-2.amazonaws.com/viz-data/total_population_dataset.parquet"
    # data_path = "../data/state-wise-population/California.parquet"
    # get file names as a list from the `../data/state-wise-population` directory and remove extension from the file names
    # file_names = [f.split(".")[0] for f in os.listdir("../data/state-wise-population")]
    # print(file_names)
    # check_dataset(census_data_url, data_path)
    # cudf DataFrame
    # c_df = load_dataset(data_path, "cudf")

    # Register top-level callback that updates plots
    register_update_plots_callback()


if __name__ == "__main__":
    # development entry point
    read_dataset()

    # Launch dashboard
    app.run_server(
        debug=True,
        dev_tools_hot_reload=True,
        dev_tools_silence_routes_logging=True,
        host="0.0.0.0",
    )
